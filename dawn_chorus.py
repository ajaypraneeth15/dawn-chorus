#!/usr/bin/env python3
"""Dawn Chorus: an offline bird-call identifier that gets you outside.

Two open-weight BirdNET models do the work, both running locally via TFLite:
  * the audio classifier (3-second windows -> ~6,500 species), and
  * the "meta" model (lat, lon, week-of-year -> which species are plausible).

Commands
  plan    Build a printable 5x5 bird-bingo card of what is likely where you
          are THIS week, favouring birds you have not heard yet.
  listen  Identify birds in a recording from your walk, write illustrated
          field notes, and add new species to your local life list.
          Optional --journal: a local Ollama model writes a short journal entry.
  companion  A short chat with a local Gemma (via Ollama) before you go out: it
          reads this week's bingo card and your life list, suggests a walk
          mission, and sends you outside. Optional, needs Ollama.
  life    Print your life list.
  doctor  Check that models load and that inference touches no network.

By default nothing here opens a network connection. The only exceptions are the
opt-in --journal flag and the `companion` command, which talk to an Ollama server
on this same machine (localhost). Your recordings, locations and life list never leave the machine.
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import html
import importlib.util
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 3
WINDOW = SAMPLE_RATE * WINDOW_SECONDS
LOCATION_THRESHOLD = 0.03  # BirdNET default for "plausible here, this week"

AUDIO_MODEL = "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
META_MODEL = "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"
LABELS = "BirdNET_GLOBAL_6K_V2.4_Labels.txt"

CITATION = (
    "Kahl, Wood, Eibl &amp; Klinck (2021). BirdNET: A deep learning solution for "
    "avian diversity monitoring. <i>Ecological Informatics</i> 61, 101236. "
    "Model weights: CC BY-NC-SA 4.0."
)


# --------------------------------------------------------------------------
# Model discovery and inference
# --------------------------------------------------------------------------
def find_models(explicit: str | None = None) -> Path:
    """Locate the BirdNET files without importing any wrapper package.

    Looks at --models, $DAWN_CHORUS_MODELS, the `birdnetlib` wheel (which ships
    the weights; we only read its files, we never import it), then ./models.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("DAWN_CHORUS_MODELS"):
        candidates.append(Path(os.environ["DAWN_CHORUS_MODELS"]))
    spec = importlib.util.find_spec("birdnetlib")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(list(spec.submodule_search_locations)[0]) / "models" / "analyzer")
    candidates.append(Path(__file__).parent / "models")
    for c in candidates:
        if all((c / f).exists() for f in (AUDIO_MODEL, META_MODEL, LABELS)):
            return c
    raise SystemExit(
        "Could not find the BirdNET model files.\n"
        "  pip install --no-deps birdnetlib      (the wheel bundles the weights)\n"
        "or point --models / $DAWN_CHORUS_MODELS at a folder containing:\n"
        f"  {AUDIO_MODEL}\n  {META_MODEL}\n  {LABELS}"
    )


def _interpreter_class():
    """Return a TFLite Interpreter class from whichever runtime is installed.

    If none can be loaded, say WHY (a missing package and a package that is installed
    but fails to load, for example a missing Windows DLL, need different fixes).
    """
    problems = []
    for module in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            return importlib.import_module(module).Interpreter
        except ImportError as e:
            problems.append(f"  {module}: {type(e).__name__}: {e}")
        except AttributeError as e:  # module loaded but has no Interpreter
            problems.append(f"  {module}: {type(e).__name__}: {e}")
    raise SystemExit("Need a TFLite runtime:  pip install ai-edge-litert\n"
                     "If that is already installed, this is why it could not be loaded:\n"
                     + "\n".join(problems))


@dataclass
class Species:
    sci: str
    common: str


class BirdNet:
    def __init__(self, models_dir: Path, threads: int = 2):
        Interpreter = _interpreter_class()
        self.dir = models_dir
        raw = (models_dir / LABELS).read_text(encoding="utf-8").splitlines()
        self.species = []
        for line in raw:
            if not line.strip():
                continue
            sci, _, common = line.partition("_")
            self.species.append(Species(sci, common or sci))
        # BirdNET also has 7 non-species classes (Engine, Dog, Siren, Gun, Noise...).
        # Real species are binomials ("Genus species"); the noise classes are one word.
        self.is_bird = np.array([" " in s.sci for s in self.species])
        self.audio = Interpreter(model_path=str(models_dir / AUDIO_MODEL), num_threads=threads)
        self.audio.allocate_tensors()
        self.meta = Interpreter(model_path=str(models_dir / META_MODEL), num_threads=1)
        self.meta.allocate_tensors()
        self._a_in = self.audio.get_input_details()[0]["index"]
        self._a_out = self.audio.get_output_details()[0]["index"]
        self._m_in = self.meta.get_input_details()[0]["index"]
        self._m_out = self.meta.get_output_details()[0]["index"]
        n_out = self.audio.get_output_details()[0]["shape"][-1]
        if n_out != len(self.species):
            raise SystemExit(f"Label file has {len(self.species)} species but model outputs {n_out}.")

    def location_scores(self, lat: float, lon: float, week: int) -> np.ndarray:
        """P(species is present) for a place and week, from the meta model."""
        x = np.array([[lat, lon, week]], dtype=np.float32)
        self.meta.set_tensor(self._m_in, x)
        self.meta.invoke()
        return self.meta.get_tensor(self._m_out)[0].copy()

    def classify(self, window: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Sigmoid scores (one per species) for one 3-second window."""
        self.audio.set_tensor(self._a_in, window[np.newaxis, :].astype(np.float32))
        self.audio.invoke()
        logits = self.audio.get_tensor(self._a_out)[0]
        return 1.0 / (1.0 + np.exp(-sensitivity * np.clip(logits, -15, 15)))


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def load_audio(path: Path) -> np.ndarray:
    """Decode anything ffmpeg can read (m4a, mp3, ogg, wav...) to 48 kHz mono."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg is required to read audio files.\n"
                         "  Windows: winget install Gyan.FFmpeg   (then close and reopen the terminal)\n"
                         "  macOS:   brew install ffmpeg      Linux: sudo apt install ffmpeg")
    if proc.returncode != 0 or not proc.stdout:
        raise SystemExit(f"ffmpeg could not read {path}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def windows(audio: np.ndarray, overlap: float):
    """Yield (start_second, 144000-sample window). Pads a final partial window
    with silence if at least one second of it is real audio."""
    hop = int((WINDOW_SECONDS - overlap) * SAMPLE_RATE)
    if hop <= 0:
        raise SystemExit("--overlap must be less than 3 seconds.")
    start = 0
    while start < len(audio):
        chunk = audio[start:start + WINDOW]
        if len(chunk) < SAMPLE_RATE:
            break
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
        yield start / SAMPLE_RATE, chunk
        start += hop


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
@dataclass
class Detection:
    sci: str
    common: str
    best: float
    place_score: float | None
    hits: list[tuple[float, float]] = field(default_factory=list)  # (start_s, conf)

    @property
    def first(self) -> float:
        return self.hits[0][0]


def week48(d: dt.date) -> int:
    """BirdNET's 48-weeks-a-year convention."""
    days = 366 if calendar.isleap(d.year) else 365
    return max(1, min(48, math.ceil(d.timetuple().tm_yday / days * 48)))


def identify(net: BirdNet, audio: np.ndarray, *, lat, lon, when: dt.date,
             min_conf: float, overlap: float, sensitivity: float,
             location_threshold: float = LOCATION_THRESHOLD):
    """Run a recording through BirdNET. Returns (detections, n_windows)."""
    place = None
    allowed = net.is_bird.copy()
    if lat is not None and lon is not None:
        place = net.location_scores(lat, lon, week48(when))
        allowed &= place >= location_threshold
    found: dict[int, Detection] = {}
    n = 0
    for start, win in windows(audio, overlap):
        n += 1
        scores = net.classify(win, sensitivity)
        for i in np.flatnonzero((scores >= min_conf) & allowed):
            sp = net.species[i]
            det = found.setdefault(int(i), Detection(
                sp.sci, sp.common, 0.0, float(place[i]) if place is not None else None))
            det.hits.append((start, float(scores[i])))
            det.best = max(det.best, float(scores[i]))
    return sorted(found.values(), key=lambda d: d.first), n


# --------------------------------------------------------------------------
# Life list (SQLite, on your disk)
# --------------------------------------------------------------------------
def default_db() -> Path:
    return Path(os.environ.get("DAWN_CHORUS_DB", Path.home() / ".dawn_chorus" / "life_list.db"))


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS sightings(
        sci TEXT NOT NULL, common TEXT NOT NULL, seen_on TEXT NOT NULL,
        lat REAL, lon REAL, place TEXT, confidence REAL, hits INTEGER,
        source TEXT NOT NULL, first_at_s REAL,
        PRIMARY KEY (sci, seen_on, source))""")
    return db


def heard_before(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT DISTINCT sci FROM sightings")}


def save_sightings(db, dets, *, when, lat, lon, place, source) -> None:
    db.executemany(
        "INSERT OR REPLACE INTO sightings VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(d.sci, d.common, when.isoformat(), lat, lon, place, d.best, len(d.hits), source, d.first)
         for d in dets])
    db.commit()


# --------------------------------------------------------------------------
# Bingo card
# --------------------------------------------------------------------------
def pick_targets(net: BirdNet, lat, lon, when, heard: set[str], n=24):
    """Choose n species likely here this week. Unheard birds first, weighted by
    how likely they are; the draw is seeded so the card is stable all week."""
    week = week48(when)
    scores = net.location_scores(lat, lon, week)
    scores = np.where(net.is_bird, scores, 0.0)
    pool = [i for i in np.flatnonzero(scores >= 0.10)]
    if len(pool) < n:
        pool = [i for i in np.flatnonzero(scores >= LOCATION_THRESHOLD)]
    seed = f"{round(lat * 2) / 2}:{round(lon * 2) / 2}:{when.year}:{week}:{len(heard)}"
    rng = random.Random(seed)

    def draw(idx, k):
        # Efraimidis-Spirakis weighted sampling without replacement.
        keyed = sorted(idx, key=lambda i: rng.random() ** (1.0 / max(float(scores[i]), 1e-6)), reverse=True)
        return keyed[:k]

    fresh = [i for i in pool if net.species[i].sci not in heard]
    chosen = draw(fresh, n)
    if len(chosen) < n:
        known = [i for i in pool if net.species[i].sci in heard]
        chosen += draw(known, n - len(chosen))
    rng.shuffle(chosen)
    return [(net.species[i], float(scores[i])) for i in chosen], len(pool)


def likelihood_dots(p: float) -> tuple[str, str]:
    if p >= 0.5:
        return "●●●", "likely"
    if p >= 0.2:
        return "●●○", "possible"
    return "●○○", "lucky find"


PAGE_CSS = """
:root{--paper:#f6f1e4;--ink:#23301f;--moss:#4d6b3c;--leaf:#8aa66b;--rust:#b4572b;--line:#cfc7ad}
*{box-sizing:border-box}
body{margin:0;background:#e9e3d0;color:var(--ink);font:16px/1.45 Georgia,'Iowan Old Style',serif}
.sheet{max-width:880px;margin:24px auto;background:var(--paper);padding:34px 38px;border:1px solid var(--line);
  box-shadow:0 2px 14px rgba(40,50,30,.18)}
h1{margin:0;font-size:34px;letter-spacing:.01em;color:var(--moss)}
.sub{margin:4px 0 18px;color:#5b6650;font-size:14px}
.sub b{color:var(--ink)}
footer{margin-top:22px;padding-top:10px;border-top:1px solid var(--line);font-size:11.5px;color:#6a735e}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:0;border:2px solid var(--moss)}
.cell{min-height:118px;padding:9px 9px 7px;border:1px solid var(--line);display:flex;flex-direction:column;position:relative}
.cell .box{width:16px;height:16px;border:2px solid var(--moss);border-radius:50%;position:absolute;top:8px;right:8px}
.cell b{font-size:15px;line-height:1.15;padding-right:22px;margin-top:2px}
.cell i{font-size:11.5px;color:#68725c;margin-top:3px}
.cell .dots{margin-top:auto;font-size:10.5px;color:var(--rust);letter-spacing:.08em}
.cell .dots span{color:#7a8470;letter-spacing:0;margin-left:4px}
.cell.free{background:repeating-linear-gradient(45deg,#ece6d2,#ece6d2 8px,#f1ecdb 8px,#f1ecdb 16px);
  align-items:center;justify-content:center;text-align:center;font-style:italic;color:var(--moss)}
.cell.free b{padding:0;font-size:14px}
.cell.known{background:#eef0e0}
.legend{margin:12px 0 0;font-size:12.5px;color:#5b6650}
.legend .d{color:var(--rust);letter-spacing:.08em}
.rules{margin:14px 0 0;font-size:13.5px}
.fields{display:flex;gap:18px;margin:14px 0 4px;font-size:13px;color:#5b6650}
.fields span{flex:1;border-bottom:1px solid var(--ink);padding-bottom:16px}
.stats{display:flex;gap:26px;margin:2px 0 20px}
.stats div{font-size:13px;color:#5b6650}
.stats strong{display:block;font-size:30px;line-height:1;color:var(--moss)}
.bird{padding:13px 0;border-top:1px solid var(--line)}
.bird h2{margin:0;font-size:19px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.bird h2 i{font-size:13px;font-weight:normal;color:#68725c}
.new{font:700 10.5px/1 Helvetica,Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;background:var(--rust);
  color:#fff;padding:4px 7px;border-radius:3px}
.meter{height:7px;background:#e3dcc4;border-radius:4px;margin:7px 0 4px;max-width:340px}
.meter u{display:block;height:100%;background:var(--moss);border-radius:4px}
.meta{font-size:12.5px;color:#5b6650}
.tl{position:relative;height:12px;background:#e7e0c8;border-radius:3px;margin-top:7px}
.tl s{position:absolute;top:0;width:3px;height:12px;background:var(--rust);border-radius:1px;text-decoration:none}
.note{margin:18px 0 0;padding:10px 12px;background:#ece6d2;border-left:3px solid var(--leaf);font-size:13px}
.journal{margin:0 0 20px;padding:14px 16px;background:#fbf8ee;border:1px solid var(--line);border-left:4px solid var(--rust)}
.journal .jh{font:700 10.5px/1.2 Helvetica,Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--rust);margin-bottom:8px}
.journal p{margin:0 0 8px;font-style:italic}
.journal .jf{font-size:11.5px;color:#6a735e;margin-top:6px}
@media print{body{background:none}.sheet{box-shadow:none;border:none;margin:0;padding:0;max-width:none}
  .cell{break-inside:avoid}.bird{break-inside:avoid}@page{margin:14mm}}
@media (max-width:640px){.sheet{padding:18px;margin:0}.cell{min-height:96px}.cell b{font-size:13px}}
"""


def render_card(targets, *, when, lat, lon, place, pool_size, heard) -> str:
    cells = []
    for k, (sp, p) in enumerate(targets):
        if k == 12:
            cells.append('<div class="cell free"><b>Free square: any bird that surprises you</b></div>')
        dots, label = likelihood_dots(p)
        known = " known" if sp.sci in heard else ""
        cells.append(
            f'<div class="cell{known}"><span class="box"></span><b>{html.escape(sp.common)}</b>'
            f'<i>{html.escape(sp.sci)}</i><div class="dots">{dots}<span>{label}</span></div></div>')
    if len(targets) <= 12:
        cells.append('<div class="cell free"><b>Free square</b></div>')
    where = html.escape(place) if place else f"{lat:.2f}, {lon:.2f}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dawn Chorus bingo</title>
<style>{PAGE_CSS}</style></head><body><div class="sheet">
<h1>Dawn Chorus Bingo</h1>
<p class="sub"><b>{where}</b> · week of {_long_date(when)} · {pool_size} species plausible here right now</p>
<div class="grid">{''.join(cells)}</div>
<p class="legend"><span class="d">●●●</span> likely &nbsp; <span class="d">●●○</span> possible &nbsp;
<span class="d">●○○</span> lucky find &nbsp;·&nbsp; shaded squares are birds already on your life list</p>
<div class="fields"><span>Date &amp; time</span><span>Where I stood</span><span>Weather</span></div>
<p class="rules">Print this, switch your phone to airplane mode, and go stand somewhere green for twenty minutes.
Tick a square when you hear or see the bird. Record a voice memo while you listen and run
<code>dawn_chorus.py listen</code> at home to check your ears against the model.</p>
<footer>Made offline by Dawn Chorus with the BirdNET location model. Likelihoods are model estimates for this
place and week, not guarantees. {CITATION}</footer></div></body></html>"""


def render_notes(dets, *, source, duration, n_windows, when, lat, lon, place, lifers, min_conf,
                 journal=None) -> str:
    where = html.escape(place) if place else (f"{lat:.2f}, {lon:.2f}" if lat is not None else "location not set")
    journal_html = ""
    if journal:
        model, text = journal
        paras = "".join(f"<p>{html.escape(p.strip())}</p>" for p in text.split("\n\n") if p.strip())
        journal_html = (f'<div class="journal"><div class="jh">Field journal · written by {html.escape(model)}, '
                        f'running locally in Ollama</div>{paras}'
                        f'<div class="jf">AI-written from the detections below. It can be wrong; the list is the record.</div></div>')
    items = []
    for d in sorted(dets, key=lambda d: (d.sci not in lifers, -d.best)):
        ticks = "".join(
            f'<s style="left:{min(99.2, t / duration * 100):.2f}%" title="{int(t // 60)}:{int(t % 60):02d} · {c:.0%}"></s>'
            for t, c in d.hits)
        when_txt = ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t, _ in d.hits[:8])
        more = f" +{len(d.hits) - 8} more" if len(d.hits) > 8 else ""
        likely = ""
        if d.place_score is not None:
            likely = f" · {likelihood_dots(d.place_score)[1]} here this week"
        badge = '<span class="new">New for your life list</span>' if d.sci in lifers else ""
        items.append(
            f'<div class="bird"><h2>{html.escape(d.common)} <i>{html.escape(d.sci)}</i> {badge}</h2>'
            f'<div class="meter"><u style="width:{d.best * 100:.0f}%"></u></div>'
            f'<div class="meta">best confidence {d.best:.0%} · {len(d.hits)} window{"s" if len(d.hits) != 1 else ""}'
            f'{likely}<br>heard at {when_txt}{more}</div><div class="tl">{ticks}</div></div>')
    body = "".join(items) or '<p class="meta">No birds cleared the confidence bar in this recording. Try a quieter spot, or lower --min-conf.</p>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Field notes</title>
<style>{PAGE_CSS}</style></head><body><div class="sheet">
<h1>Field notes</h1>
<p class="sub"><b>{where}</b> · {_long_date(when, weekday=True)} · <code>{html.escape(source)}</code></p>
<div class="stats"><div><strong>{len(dets)}</strong>species</div><div><strong>{len(lifers)}</strong>new for your life list</div>
<div><strong>{duration / 60:.1f}</strong>minutes listened</div></div>
{journal_html}{body}
<p class="note">BirdNET is a strong listener, not an infallible one. Anything you are keeping, confirm by ear or by
eye. Detections were kept at confidence ≥ {min_conf:.0%} across {n_windows} three-second windows.</p>
<footer>Everything in this file was computed on your own machine. {CITATION}</footer></div></body></html>"""


# --------------------------------------------------------------------------
# Optional: a field-journal paragraph from a local Ollama model
# --------------------------------------------------------------------------
OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(Exception):
    """Raised with a human-readable reason when the journal can't be written."""


def journal_prompt(dets, *, place, when) -> str:
    facts = "\n".join(
        f"- {d.common} ({d.sci}): heard in {len(d.hits)} three-second window{'s' if len(d.hits) != 1 else ''}, "
        f"first at {_fmt_t(d.first)}, confidence {d.best:.0%}" for d in dets)
    return (
        "You write short, warm field-journal entries for a hobby birdwatcher.\n"
        "Use ONLY the facts below. Do not mention any bird, place, weather, habitat, or event that is not "
        "listed, and do not invent behaviour. Write 3 to 4 sentences of plain prose: no lists, no headings.\n\n"
        f"Place: {place or 'not given'}\n"
        f"Date: {_long_date(when, weekday=True)}\n"
        "Birds heard (automatic detections, may contain errors):\n"
        f"{facts}\n\nJournal entry:")


def _ollama_client(host):
    """Return (host, call). `call(path, payload=None, t=10)` does GET (no payload) or POST (JSON) and parses JSON.

    Uses only the standard library. A server on this computer is never reached through a proxy.
    """
    import urllib.request
    host = host.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if _is_local(host) \
        else urllib.request.build_opener()

    def call(path, payload=None, t=10):
        req = urllib.request.Request(
            host + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=t) as resp:
            return json.load(resp)

    return host, call


@contextlib.contextmanager
def _ollama_errors(host, current_model=lambda: None):
    """Turn network/HTTP/JSON failures into an OllamaError with a human-readable reason."""
    import urllib.error
    try:
        yield
    except OllamaError:
        raise
    except urllib.error.HTTPError as e:
        model = current_model()
        hint = f"  Is '{model}' installed? Try:  ollama pull {model}" if e.code == 404 and model else ""
        raise OllamaError(f"Ollama answered with an error ({e.code}).{hint}") from e
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"Could not reach Ollama at {host}. Is it running? (open the Ollama app, or run: "
                          f"ollama serve)") from e
    except ValueError as e:  # bad JSON
        raise OllamaError("Ollama sent a reply this tool could not read.") from e


def _pick_model(call, model):
    """The model you named, else the first installed Gemma, else the first installed model."""
    if model:
        return model
    names = [m.get("name") for m in call("/api/tags").get("models", []) if m.get("name")]
    if not names:
        raise OllamaError("Ollama is running but has no models yet. Try:  ollama pull llama3.2")
    return next((n for n in names if "gemma" in n.lower()), names[0])  # prefer Gemma if installed


def ollama_journal(dets, *, place, when, model=None, host=OLLAMA_DEFAULT_HOST, timeout=180):
    """Ask an Ollama server for a short journal entry. Returns (model_name, text).

    Uses only the standard library. The default host is this machine, so the
    detections never leave it. Raises OllamaError with a friendly reason.
    """
    host, call = _ollama_client(host)
    with _ollama_errors(host, lambda: model):
        model = _pick_model(call, model)
        out = call("/api/generate", {
            "model": model, "prompt": journal_prompt(dets, place=place, when=when), "stream": False,
            "options": {"temperature": 0.4, "num_predict": 260}}, t=timeout)
    text = (out.get("response") or "").strip()
    if not text:
        raise OllamaError(f"Model '{model}' returned an empty answer.")
    return model, text


def ollama_chat(messages, *, model=None, host=OLLAMA_DEFAULT_HOST, timeout=180):
    """One turn of a conversation with a local Ollama model (POST /api/chat). Returns (model_name, reply).

    `messages` is the whole history so far: [{"role": "system"|"user"|"assistant", "content": str}, ...].
    """
    host, call = _ollama_client(host)
    with _ollama_errors(host, lambda: model):
        model = _pick_model(call, model)
        out = call("/api/chat", {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.5, "num_predict": 200}}, t=timeout)
    if not isinstance(out, dict):
        raise OllamaError("Ollama sent a reply this tool could not read.")
    text = ((out.get("message") or {}).get("content") or "").strip()
    if not text:
        raise OllamaError(f"Model '{model}' returned an empty answer.")
    return model, text


# --------------------------------------------------------------------------
# Optional: the walk companion (a short chat with a local Gemma, before you go out)
# --------------------------------------------------------------------------
MISSION_REQUEST = "I'm about to go for a walk. Give me a short mission for it, using my bingo card."
END_WORDS = {"", "go", "bye", "quit", "exit", "q", "ok", "okay"}


def companion_system_prompt(targets, *, place, when, life_count, recent, language=None) -> str:
    """Ground the model in facts we really have, and tell it its job is to send the person outside."""
    card = "\n".join(f"- {sp.common}: {likelihood_dots(score)[1]} this week" for sp, score in targets)
    recent_txt = ", ".join(recent) if recent else "none yet"
    lang = f"Reply in {language}. " if language else ""
    return (
        "You are Dawn Chorus, a friendly outdoor companion for a beginner birdwatcher. Your job is to get "
        "them OUTSIDE and away from the screen, not to keep them chatting.\n"
        f"{lang}"
        "Rules:\n"
        "- Reply in at most 3 short sentences, in warm, plain language.\n"
        "- Always point toward going outside. When the person says they are ready, or after a few messages, "
        "tell them to put the phone away.\n"
        "- About birds you may mention ONLY the birds in the bingo card below, and only how likely each is "
        "this week as listed. Do not describe their songs, looks, behaviour or habitat. If asked about a bird "
        "and you are not sure, say so and suggest a field guide.\n"
        "- Safe general tips are fine: stand still and listen for five quiet minutes, listen before looking, "
        "look up and along tree edges, carry water, tell someone where you are going.\n"
        "- Do not invent weather, places, or events.\n\n"
        f"Today: {_long_date(when, weekday=True)} (week {week48(when)} of 48). Place: {place or 'not given'}.\n"
        "This week's bingo card (likelihood comes from the BirdNET location model):\n"
        f"{card}\n"
        f"Life list: {life_count} species. Most recently added: {recent_txt}.")


def companion_session(chat, system, *, turns, once, ask, say):
    """Run the chat. `chat(messages) -> reply text`; `ask(prompt) -> str`; `say(text)`.

    Short on purpose: one opening mission, then at most `turns` replies, ending when the person
    sends an empty line or says go/bye. Returns the message history.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": MISSION_REQUEST}]

    def turn():
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        say(reply)

    turn()
    if once:
        return messages
    for _ in range(max(0, turns)):
        try:
            text = ask("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in END_WORDS:
            break
        messages.append({"role": "user", "content": text})
        try:
            turn()
        except OllamaError as e:
            messages.pop()
            say(f"(the companion lost its connection: {e})")
            break
    return messages


def _is_local(host: str) -> bool:
    from urllib.parse import urlparse
    return (urlparse(host).hostname or "") in ("127.0.0.1", "localhost", "::1")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _long_date(d: dt.date, weekday: bool = False) -> str:
    """'October 6, 2026' / 'Tuesday, October 6, 2026'. Avoids strftime's '%-d', which Windows rejects."""
    text = f"{d.strftime('%B')} {d.day}, {d.year}"
    return f"{d.strftime('%A')}, {text}" if weekday else text


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _coords(args):
    lat = args.lat if args.lat is not None else _env_float("DAWN_CHORUS_LAT")
    lon = args.lon if args.lon is not None else _env_float("DAWN_CHORUS_LON")
    if (lat is None) != (lon is None):
        raise SystemExit("Give both --lat and --lon (or neither).")
    if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise SystemExit("Latitude must be -90..90 and longitude -180..180.")
    return lat, lon


def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v else None


def _date(args) -> dt.date:
    return dt.date.fromisoformat(args.date) if args.date else dt.date.today()


def cmd_plan(args):
    lat, lon = _coords(args)
    if lat is None:
        raise SystemExit("plan needs a location: --lat/--lon (or $DAWN_CHORUS_LAT / $DAWN_CHORUS_LON).")
    when = _date(args)
    net = BirdNet(find_models(args.models))
    db = open_db(Path(args.db) if args.db else default_db())
    heard = heard_before(db)
    targets, pool = pick_targets(net, lat, lon, when, heard, n=24)
    Path(args.out).write_text(
        render_card(targets, when=when, lat=lat, lon=lon, place=args.place,
                    pool_size=pool, heard=heard), encoding="utf-8")
    db.close()
    fresh = sum(1 for sp, _ in targets if sp.sci not in heard)
    print(f"Wrote {args.out}: 24 targets ({fresh} you haven't heard yet), {pool} species plausible "
          f"at {lat:.2f},{lon:.2f} in week {week48(when)}/48.")
    print("Print it, go outside, leave the phone in your pocket.")


def cmd_listen(args):
    lat, lon = _coords(args)
    when = _date(args)
    net = BirdNet(find_models(args.models))
    db = None if args.no_save else open_db(Path(args.db) if args.db else default_db())
    known = heard_before(db) if db else set()
    for path in args.audio:
        path = Path(path)
        t0 = time.time()
        audio = load_audio(path)
        dur = len(audio) / SAMPLE_RATE
        dets, n = identify(net, audio, lat=lat, lon=lon, when=when, min_conf=args.min_conf,
                           overlap=args.overlap, sensitivity=args.sensitivity)
        lifers = {d.sci for d in dets if d.sci not in known}
        print(f"\n{path.name}: {dur:.0f}s, {n} windows, {time.time() - t0:.1f}s on this machine"
              + ("" if lat is not None else "  (no location given: nothing filtered by place)"))
        for d in sorted(dets, key=lambda d: -d.best):
            tag = "  ★ NEW" if d.sci in lifers else ""
            print(f"  {d.best:5.0%}  {d.common:<32} x{len(d.hits):<3} first at {_fmt_t(d.first)}{tag}")
        if not dets:
            print("  (no detections above the confidence bar)")
        journal = None
        if args.journal and dets:
            if not _is_local(args.ollama_host):
                print(f"  NOTE: {args.ollama_host} is not this machine. Your detections will be sent there.")
            try:
                journal = ollama_journal(dets, place=args.place, when=when, model=args.ollama_model,
                                         host=args.ollama_host)
                print(f"  journal written by {journal[0]} (local Ollama)")
            except OllamaError as e:
                print(f"  journal skipped: {e}")
        out = Path(args.out) if args.out and len(args.audio) == 1 else path.with_suffix(".notes.html")
        out.write_text(render_notes(dets, source=path.name, duration=max(dur, 1), n_windows=n, when=when,
                                    lat=lat, lon=lon, place=args.place, lifers=lifers,
                                    min_conf=args.min_conf, journal=journal), encoding="utf-8")
        print(f"  field notes -> {out}")
        if db:
            save_sightings(db, dets, when=when, lat=lat, lon=lon, place=args.place, source=path.name)
            known |= {d.sci for d in dets}
    if db:
        print(f"\nLife list now holds {len(heard_before(db))} species.")
        db.close()


def cmd_companion(args):
    lat, lon = _coords(args)
    if lat is None:
        raise SystemExit("companion needs a location: --lat/--lon (or $DAWN_CHORUS_LAT / $DAWN_CHORUS_LON).")
    when = _date(args)
    net = BirdNet(find_models(args.models))
    db = open_db(Path(args.db) if args.db else default_db())
    heard = heard_before(db)
    recent = [r[0] for r in db.execute(
        "SELECT common FROM sightings GROUP BY sci ORDER BY MIN(seen_on) DESC, common LIMIT 5")]
    db.close()
    targets, _ = pick_targets(net, lat, lon, when, heard, n=24)
    system = companion_system_prompt(targets, place=args.place, when=when, life_count=len(heard),
                                     recent=recent, language=args.language)
    if not _is_local(args.ollama_host):
        print(f"  NOTE: {args.ollama_host} is not this machine. Your bingo birds and life list names will be sent there.")
    state = {"model": args.ollama_model}

    def chat(messages):
        state["model"], reply = ollama_chat(messages, model=state["model"], host=args.ollama_host)
        return reply

    print("Dawn Chorus companion. Press Enter (or type 'go') whenever you are ready to head outside.\n")
    try:
        companion_session(chat, system, turns=min(max(args.turns, 0), 10), once=args.once, ask=input, say=lambda text: print(text + "\n"))
    except OllamaError as e:
        raise SystemExit(f"companion unavailable: {e}\n"
                         f"Your bingo card still works without it:  python dawn_chorus.py plan --lat {lat} --lon {lon}")
    print(f"\n  (chatted with {state['model']}, running on this computer)" if _is_local(args.ollama_host) else
          f"\n  (chatted with {state['model']} at {args.ollama_host})")
    print("Time to go outside. Phone in your pocket, five quiet minutes, tick birds in pencil.")
    print(f"When you are back:  python dawn_chorus.py listen <your recording> --lat {lat} --lon {lon} --journal")


def cmd_life(args):
    db = open_db(Path(args.db) if args.db else default_db())
    rows = db.execute("""SELECT common, sci, MIN(seen_on), COUNT(DISTINCT seen_on), MAX(confidence)
                         FROM sightings GROUP BY sci ORDER BY MIN(seen_on), common""").fetchall()
    print(f"Life list: {len(rows)} species   ({default_db() if not args.db else args.db})")
    for common, sci, first, days, conf in rows:
        print(f"  {first}  {common:<32} {days} day{'s' if days != 1 else ' '}  best {conf:.0%}")
    db.close()


def cmd_doctor(args):
    t0 = time.time()
    d = find_models(args.models)
    net = BirdNet(d)
    load = time.time() - t0
    t1 = time.time()
    net.location_scores(42.45, -76.5, week48(dt.date.today()))
    net.classify(np.zeros(WINDOW, dtype=np.float32))
    run = time.time() - t1
    print(f"models dir : {d}")
    for f in (AUDIO_MODEL, META_MODEL, LABELS):
        print(f"  {f}  {(d / f).stat().st_size / 1e6:.1f} MB")
    print(f"species    : {len(net.species)}")
    print(f"runtime    : {_interpreter_class().__module__}")
    print(f"load {load:.2f}s, one window + location lookup {run * 1000:.0f} ms")
    net_mods = sorted(m for m in ("socket", "ssl", "http.client", "urllib.request", "requests") if m in sys.modules)
    print("network modules imported:", ", ".join(net_mods) if net_mods else "none")


def build_parser():
    p = argparse.ArgumentParser(prog="dawn_chorus", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", help="folder with the BirdNET .tflite + labels files")
    p.add_argument("--db", help="life list database (default ~/.dawn_chorus/life_list.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def loc(sp):
        sp.add_argument("--lat", type=float)
        sp.add_argument("--lon", type=float)
        sp.add_argument("--date", help="YYYY-MM-DD (default: today)")
        sp.add_argument("--place", help="name for the page header, e.g. 'Beebe Lake'")

    sp = sub.add_parser("plan", help="print-ready bird bingo for this week, here")
    loc(sp); sp.add_argument("--out", default="bingo.html"); sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("listen", help="identify birds in recording(s) and keep a life list")
    sp.add_argument("audio", nargs="+"); loc(sp)
    sp.add_argument("--min-conf", type=float, default=0.5, help="confidence needed to count (default 0.5)")
    sp.add_argument("--overlap", type=float, default=0.0, help="seconds of window overlap, 0-2.9 (default 0)")
    sp.add_argument("--sensitivity", type=float, default=1.0)
    sp.add_argument("--no-save", action="store_true", help="don't touch the life list")
    sp.add_argument("--journal", action="store_true",
                    help="add a short journal entry written by a local Ollama model (optional)")
    sp.add_argument("--ollama-model", help="Ollama model name, e.g. a Gemma model (default: a Gemma model if you have one, else your first model)")
    sp.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST,
                    help=f"Ollama server address (default {OLLAMA_DEFAULT_HOST}, i.e. this computer)")
    sp.add_argument("--out", help="field notes path (single file only)"); sp.set_defaults(fn=cmd_listen)

    sp = sub.add_parser("companion", help="optional: a short chat with a local Gemma that sends you outside")
    loc(sp)
    sp.add_argument("--turns", type=int, default=4, help="how many of your messages to answer (default 4, max 10)")
    sp.add_argument("--once", action="store_true", help="just print the walk mission and exit")
    sp.add_argument("--language", help="ask the companion to reply in this language, e.g. Tamil (quality depends on the model)")
    sp.add_argument("--ollama-model", help="Ollama model name (default: a Gemma model if you have one, else your first model)")
    sp.add_argument("--ollama-host", default=OLLAMA_DEFAULT_HOST,
                    help=f"Ollama server address (default {OLLAMA_DEFAULT_HOST}, i.e. this computer)")
    sp.set_defaults(fn=cmd_companion)

    sp = sub.add_parser("life", help="print your life list"); sp.set_defaults(fn=cmd_life)
    sp = sub.add_parser("doctor", help="verify models load and nothing needs the network")
    sp.set_defaults(fn=cmd_doctor)
    return p


def _console_safe():
    """Never crash just because the console cannot show a character (e.g. the star on some Windows consoles)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _console_safe()
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
