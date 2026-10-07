"""Run with:  python3 -m unittest discover -s tests -v

Unit tests need only numpy. Model/audio tests run when the BirdNET weights are
installed (pip install --no-deps birdnetlib) and, for the end-to-end test, when
examples/soundscape.wav exists (scripts/get_example_audio.sh).
"""
import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dawn_chorus as dc  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "soundscape.wav"
ITHACA = (42.45, -76.50)


def _models():
    try:
        return dc.find_models()
    except SystemExit:
        return None


MODELS = _models()


class PureLogic(unittest.TestCase):
    def test_week48_bounds_and_known_dates(self):
        self.assertEqual(dc.week48(dt.date(2026, 1, 1)), 1)
        self.assertEqual(dc.week48(dt.date(2026, 12, 31)), 48)
        self.assertEqual(dc.week48(dt.date(2026, 10, 6)), 37)
        self.assertEqual(dc.week48(dt.date(2028, 12, 31)), 48)  # leap year

    def test_windows_pad_tail_and_drop_scraps(self):
        sr = dc.SAMPLE_RATE
        import numpy as np
        # 7.5 s -> windows at 0, 3, 6 (last has 1.5 s real audio, padded)
        got = list(dc.windows(np.ones(int(7.5 * sr), dtype=np.float32), overlap=0))
        self.assertEqual([round(s) for s, _ in got], [0, 3, 6])
        self.assertTrue(all(len(w) == dc.WINDOW for _, w in got))
        self.assertEqual(float(got[-1][1][-1]), 0.0)
        # 6.5 s -> trailing 0.5 s is a scrap and is dropped
        self.assertEqual(len(list(dc.windows(np.ones(int(6.5 * sr), dtype=np.float32), 0))), 2)

    def test_overlap_shrinks_hop(self):
        import numpy as np
        a = np.ones(9 * dc.SAMPLE_RATE, dtype=np.float32)
        self.assertEqual([round(s, 1) for s, _ in dc.windows(a, overlap=1.0)][:4], [0.0, 2.0, 4.0, 6.0])
        with self.assertRaises(SystemExit):
            list(dc.windows(a, overlap=3.0))

    def test_life_list_is_idempotent_and_tracks_first_sighting(self):
        with tempfile.TemporaryDirectory() as d:
            db = dc.open_db(Path(d) / "l.db")
            det = dc.Detection("Poecile atricapillus", "Black-capped Chickadee", 0.8, 0.9, [(3.0, 0.8)])
            kw = dict(lat=1.0, lon=2.0, place="x", source="a.wav")
            dc.save_sightings(db, [det], when=dt.date(2026, 10, 6), **kw)
            dc.save_sightings(db, [det], when=dt.date(2026, 10, 6), **kw)  # same file again
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0], 1)
            dc.save_sightings(db, [det], when=dt.date(2026, 10, 7), **kw)  # new day
            self.assertEqual(dc.heard_before(db), {"Poecile atricapillus"})
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0], 2)
            db.close()

    def test_html_escapes_names(self):
        sp = dc.Species("A b", "<script>alert(1)</script>")
        page = dc.render_card([(sp, 0.9)], when=dt.date(2026, 10, 6), lat=1, lon=2,
                              place="<b>x</b>", pool_size=1, heard=set())
        self.assertNotIn("<script>", page)
        self.assertNotIn("<b>x</b>", page)


@unittest.skipUnless(MODELS, "BirdNET weights not installed")
class WithModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = dc.BirdNet(MODELS)

    def test_label_set_and_noise_classes(self):
        self.assertEqual(len(self.net.species), 6522)
        noise = {s.sci for s, b in zip(self.net.species, self.net.is_bird) if not b}
        self.assertEqual(noise, {"Dog", "Engine", "Environmental", "Fireworks", "Gun", "Noise", "Siren"})

    def test_location_model_knows_geography(self):
        week = dc.week48(dt.date(2026, 10, 6))
        ny = self.net.location_scores(*ITHACA, week)
        sydney = self.net.location_scores(-33.87, 151.21, week)
        idx = {s.common: i for i, s in enumerate(self.net.species)}
        self.assertGreater(ny[idx["Black-capped Chickadee"]], 0.5)
        self.assertLess(sydney[idx["Black-capped Chickadee"]], 0.03)
        self.assertGreater(sydney[idx["Australian Magpie"]], 0.5)

    def test_silence_yields_no_birds(self):
        import numpy as np
        dets, n = dc.identify(self.net, np.zeros(10 * dc.SAMPLE_RATE, dtype=np.float32),
                              lat=None, lon=None, when=dt.date(2026, 10, 6),
                              min_conf=0.5, overlap=0, sensitivity=1.0)
        self.assertEqual((dets, n), ([], 4))

    def test_bingo_is_stable_excludes_heard_and_noise(self):
        when = dt.date(2026, 10, 6)
        a, _ = dc.pick_targets(self.net, *ITHACA, when, set())
        b, _ = dc.pick_targets(self.net, *ITHACA, when, set())
        self.assertEqual([s.sci for s, _ in a], [s.sci for s, _ in b])
        self.assertEqual(len(a), 24)
        self.assertTrue(all(" " in s.sci for s, _ in a))
        heard = {s.sci for s, _ in a[:5]}
        c, _ = dc.pick_targets(self.net, *ITHACA, when, heard)
        self.assertFalse(heard & {s.sci for s, _ in c})

    def test_bingo_changes_with_place(self):
        when = dt.date(2026, 10, 6)
        ny, _ = dc.pick_targets(self.net, *ITHACA, when, set())
        syd, _ = dc.pick_targets(self.net, -33.87, 151.21, when, set())
        self.assertLess(len({s.sci for s, _ in ny} & {s.sci for s, _ in syd}), 3)

    @unittest.skipUnless(EXAMPLE.exists(), "run scripts/get_example_audio.sh first")
    def test_real_soundscape_end_to_end(self):
        audio = dc.load_audio(EXAMPLE)
        dets, n = dc.identify(self.net, audio, lat=ITHACA[0], lon=ITHACA[1], when=dt.date(2026, 10, 6),
                              min_conf=0.5, overlap=0, sensitivity=1.0)
        names = {d.common for d in dets}
        self.assertEqual(n, 40)
        self.assertTrue({"Black-capped Chickadee", "Dark-eyed Junco", "House Finch"} <= names, names)
        self.assertNotIn("Engine", names)
        # and the location filter must remove the same recording's far-away false positives
        far, _ = dc.identify(self.net, audio, lat=-33.87, lon=151.21, when=dt.date(2026, 10, 6),
                             min_conf=0.5, overlap=0, sensitivity=1.0)
        self.assertNotIn("Black-capped Chickadee", {d.common for d in far})


class FakeOllama:
    """A stand-in Ollama server on localhost speaking the two endpoints we use.
    It lets us test our client code; it is NOT a real language model."""

    def __init__(self, models=("llama3.2:latest", "gemma3:1b"), reply="A calm walk. Chickadees called.",
                 generate_status=200):
        import http.server
        import json
        import threading
        outer = self
        self.models, self.reply, self.generate_status = list(models), reply, generate_status
        self.requests = []
        self.paths = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output quiet
                pass

            def _send(self, code, payload):
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.paths.append(self.path)
                if outer.generate_status != 200:
                    self._send(outer.generate_status, {"error": "model not found"})
                elif self.path == "/api/chat":
                    self._send(200, {"model": body["model"], "done": True,
                                     "message": {"role": "assistant", "content": outer.reply}})
                else:
                    self._send(200, {"model": body["model"], "response": outer.reply, "done": True})

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _dets():
    return [dc.Detection("Poecile atricapillus", "Black-capped Chickadee", 0.81, 0.9, [(0.0, 0.81)]),
            dc.Detection("Junco hyemalis", "Dark-eyed Junco", 0.74, 0.5, [(42.0, 0.74), (60.0, 0.7)])]


class OllamaJournal(unittest.TestCase):
    when = dt.date(2026, 10, 6)

    def test_named_model_is_used_and_prompt_contains_only_detections(self):
        with FakeOllama() as srv:
            model, text = dc.ollama_journal(_dets(), place="Test trail", when=self.when,
                                            model="gemma3:1b", host=srv.url)
        self.assertEqual((model, text), ("gemma3:1b", "A calm walk. Chickadees called."))
        req = srv.requests[0]
        self.assertEqual(req["model"], "gemma3:1b")
        self.assertFalse(req["stream"])
        for needle in ("Black-capped Chickadee", "Dark-eyed Junco", "Test trail", "Use ONLY the facts below"):
            self.assertIn(needle, req["prompt"])

    def test_prefers_gemma_when_no_model_is_named(self):
        with FakeOllama(models=("llama3.2:latest", "gemma3:4b", "mistral")) as srv:
            model, _ = dc.ollama_journal(_dets(), place=None, when=self.when, host=srv.url)
        self.assertEqual(model, "gemma3:4b")

    def test_falls_back_to_first_model_without_gemma(self):
        with FakeOllama(models=("llama3.2:latest", "mistral")) as srv:
            model, _ = dc.ollama_journal(_dets(), place=None, when=self.when, host=srv.url)
        self.assertEqual(model, "llama3.2:latest")

    def test_no_models_installed_gives_friendly_error(self):
        with FakeOllama(models=()) as srv:
            with self.assertRaisesRegex(dc.OllamaError, "no models"):
                dc.ollama_journal(_dets(), place=None, when=self.when, host=srv.url)

    def test_unknown_model_gives_pull_hint(self):
        with FakeOllama(generate_status=404) as srv:
            with self.assertRaisesRegex(dc.OllamaError, "ollama pull nope"):
                dc.ollama_journal(_dets(), place=None, when=self.when, model="nope", host=srv.url)

    def test_empty_answer_is_an_error(self):
        with FakeOllama(reply="   ") as srv:
            with self.assertRaisesRegex(dc.OllamaError, "empty"):
                dc.ollama_journal(_dets(), place=None, when=self.when, model="gemma3:1b", host=srv.url)

    def test_unreachable_server_is_an_error_not_a_crash(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing is listening on this port now
        with self.assertRaisesRegex(dc.OllamaError, "Could not reach Ollama"):
            dc.ollama_journal(_dets(), place=None, when=self.when, host=f"http://127.0.0.1:{port}", timeout=3)

    def test_is_local(self):
        self.assertTrue(dc._is_local("http://127.0.0.1:11434"))
        self.assertTrue(dc._is_local("http://localhost:11434"))
        self.assertFalse(dc._is_local("http://192.168.1.20:11434"))
        self.assertFalse(dc._is_local("https://example.com"))

    def test_journal_text_is_html_escaped(self):
        page = dc.render_notes(_dets(), source="a.wav", duration=60, n_windows=20, when=self.when,
                               lat=None, lon=None, place=None, lifers=set(), min_conf=0.5,
                               journal=("gemma3:1b", "<script>alert(1)</script> Nice walk."))
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("gemma3:1b", page)
        self.assertIn("running locally in Ollama", page)

    def test_no_journal_block_by_default(self):
        page = dc.render_notes(_dets(), source="a.wav", duration=60, n_windows=20, when=self.when,
                               lat=None, lon=None, place=None, lifers=set(), min_conf=0.5)
        self.assertNotIn("Field journal", page)

    @unittest.skipUnless(MODELS and EXAMPLE.exists(), "needs BirdNET weights and examples/soundscape.wav")
    def test_cli_end_to_end_with_journal_and_with_ollama_down(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d, FakeOllama(reply="Chickadee and junco today.") as srv:
            out = Path(d) / "notes.html"
            argv = ["--db", str(Path(d) / "l.db"), "listen", str(EXAMPLE), "--lat", "42.45", "--lon", "-76.5",
                    "--date", "2026-10-06", "--no-save", "--out", str(out)]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dc.main(argv + ["--journal", "--ollama-host", srv.url])
            self.assertIn("journal written by gemma3:1b", buf.getvalue())  # Gemma preferred automatically
            self.assertIn("Chickadee and junco today.", out.read_text(encoding="utf-8"))
            self.assertIn("Black-capped Chickadee", srv.requests[0]["prompt"])
            # With Ollama unreachable the notes must still be written.
            import socket
            s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
            out2 = Path(d) / "notes2.html"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dc.main(argv[:-2] + ["--out", str(out2), "--journal", "--ollama-host", f"http://127.0.0.1:{port}"])
            self.assertIn("journal skipped", buf.getvalue())
            self.assertIn("Black-capped Chickadee", out2.read_text(encoding="utf-8"))
            self.assertNotIn("Field journal", out2.read_text(encoding="utf-8"))


class RuntimeLoadingMessage(unittest.TestCase):
    """If no TFLite runtime loads, the error must say why (e.g. a missing DLL), not just 'install it'."""

    def test_error_names_the_real_import_failure(self):
        from unittest import mock
        real_import = __import__("importlib").import_module

        def fake(name, *a, **k):
            if name == "ai_edge_litert.interpreter":
                raise ImportError("DLL load failed while importing _pywrap_litert_interpreter_wrapper")
            if name in ("tflite_runtime.interpreter", "tensorflow.lite"):
                raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
            return real_import(name, *a, **k)

        with mock.patch("importlib.import_module", side_effect=fake):
            with self.assertRaises(SystemExit) as ctx:
                dc._interpreter_class()
        msg = str(ctx.exception)
        self.assertIn("pip install ai-edge-litert", msg)
        self.assertIn("DLL load failed", msg)
        self.assertIn("tflite_runtime.interpreter", msg)


class WindowsPortability(unittest.TestCase):
    """Things that work on Linux/macOS but crash on Windows, caught here on any platform."""

    def test_no_platform_specific_strftime_codes(self):
        import re
        src = (Path(__file__).resolve().parent.parent / "dawn_chorus.py").read_text(encoding="utf-8")
        bad = re.findall(r"strftime\(\s*[fF]?['\"][^'\"]*%[-#]", src)
        self.assertEqual(bad, [], "'%-d' style codes raise ValueError on Windows")

    def test_dates_render_when_strftime_rejects_dash_codes(self):
        class WindowsLikeDate(dt.date):
            def strftime(self, fmt):
                if "%-" in fmt:
                    raise ValueError("Invalid format string")
                return super().strftime(fmt)

        when = WindowsLikeDate(2026, 10, 6)
        self.assertEqual(dc._long_date(when), "October 6, 2026")
        self.assertEqual(dc._long_date(when, weekday=True), "Tuesday, October 6, 2026")
        card = dc.render_card([], when=when, lat=42.45, lon=-76.5, place="x", pool_size=0, heard=set())
        self.assertIn("October 6, 2026", card)
        det = [dc.Detection("Poecile atricapillus", "Black-capped Chickadee", 0.8, 0.9, [(0.0, 0.8)])]
        notes = dc.render_notes(det, source="a.wav", duration=60, n_windows=20, when=when, lat=None, lon=None,
                                place=None, lifers=set(), min_conf=0.5, journal=None)
        self.assertIn("Tuesday, October 6, 2026", notes)
        self.assertIn("Tuesday, October 6, 2026", dc.journal_prompt(det, place=None, when=when))

    def test_console_that_cannot_show_a_star_does_not_crash(self):
        import io
        from unittest import mock
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with self.assertRaises(UnicodeEncodeError):
            print("\u2605 NEW", file=stream)
        with mock.patch("sys.stdout", stream), mock.patch("sys.stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252")):
            dc._console_safe()
            print("\u2605 NEW")  # must not raise


if __name__ == "__main__":
    unittest.main()
