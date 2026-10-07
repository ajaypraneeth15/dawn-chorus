"""Tests for the optional Gemma/Ollama journal, using unittest.mock only.

Run with:  python3 -m unittest discover -s tests -v

Nothing here needs Ollama, a GPU, a model download, BirdNET, or a network. The HTTP layer
(urllib's opener) is replaced by a mock, and a guard makes any real socket connection
fail the test. These tests prove our client code handles Ollama's API correctly; they say
nothing about the quality of text a real Gemma model would write.
"""
import datetime as dt
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dawn_chorus as dc  # noqa: E402

WHEN = dt.date(2026, 10, 6)


class FakeResponse:
    """What urllib hands back: a context manager with .read()."""

    def __init__(self, payload=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self, *args):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError("http://localhost:11434/x", code, "err", None, io.BytesIO(b""))


def make_opener(tags=None, generate=None):
    """A mock urllib opener that answers GET /api/tags and POST /api/generate (or /api/chat).

    `tags` / `generate` may be a payload dict (success), raw bytes (garbled body),
    or an Exception instance (failure).
    """
    def answer(spec):
        if isinstance(spec, Exception):
            raise spec
        if isinstance(spec, bytes):
            return FakeResponse(raw=spec)
        return FakeResponse(spec)

    def open_(req, timeout=None):
        if req.full_url.endswith("/api/tags"):
            return answer(tags)
        if req.full_url.endswith(("/api/generate", "/api/chat")):
            return answer(generate)
        raise AssertionError(f"unexpected URL {req.full_url}")

    opener = mock.Mock()
    opener.open.side_effect = open_
    return opener


def tags_payload(*names):
    return {"models": [{"name": n} for n in names]}


def dets():
    return [dc.Detection("Poecile atricapillus", "Black-capped Chickadee", 0.81, 0.9, [(0.0, 0.81)]),
            dc.Detection("Junco hyemalis", "Dark-eyed Junco", 0.74, 0.5, [(42.0, 0.74), (60.0, 0.7)])]


def run_journal(opener, **kw):
    kw.setdefault("place", "Test trail")
    kw.setdefault("when", WHEN)
    with mock.patch("urllib.request.build_opener", return_value=opener):
        return dc.ollama_journal(dets(), **kw)


class NoRealNetwork(unittest.TestCase):
    """Base class: any attempt to open a real socket fails the test loudly."""

    def setUp(self):
        guard = mock.patch("socket.socket.connect", side_effect=AssertionError("real network call attempted"))
        guard.start()
        self.addCleanup(guard.stop)


class SuccessPath(NoRealNetwork):
    def test_posts_to_api_generate_with_expected_payload(self):
        opener = make_opener(generate={"response": "A quiet morning with chickadees."})
        model, text = run_journal(opener, model="gemma3:1b")
        self.assertEqual((model, text), ("gemma3:1b", "A quiet morning with chickadees."))
        self.assertEqual(opener.open.call_count, 1)  # model named, so /api/tags is not queried
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/generate")
        body = json.loads(req.data)
        self.assertEqual(body["model"], "gemma3:1b")
        self.assertIs(body["stream"], False)
        self.assertIn("Black-capped Chickadee", body["prompt"])
        self.assertIn("Dark-eyed Junco", body["prompt"])
        self.assertIn("Use ONLY the facts below", body["prompt"])

    def test_response_is_stripped(self):
        opener = make_opener(generate={"response": "  hello \n"})
        self.assertEqual(run_journal(opener, model="gemma3:1b")[1], "hello")

    def test_custom_host_is_respected(self):
        opener = make_opener(generate={"response": "ok"})
        run_journal(opener, model="gemma3:1b", host="http://127.0.0.1:9999/")
        self.assertEqual(opener.open.call_args.args[0].full_url, "http://127.0.0.1:9999/api/generate")


class GemmaPriority(NoRealNetwork):
    def used_model(self, *installed, **kw):
        opener = make_opener(tags=tags_payload(*installed), generate={"response": "ok"})
        model, _ = run_journal(opener, **kw)
        generate_req = opener.open.call_args.args[0]
        self.assertEqual(json.loads(generate_req.data)["model"], model)
        return model

    def test_gemma_wins_over_models_listed_before_it(self):
        self.assertEqual(self.used_model("llama3.2:latest", "gemma3:4b", "mistral"), "gemma3:4b")

    def test_gemma_match_ignores_case(self):
        self.assertEqual(self.used_model("llama3.2", "Gemma2:2b"), "Gemma2:2b")

    def test_first_gemma_is_used_when_several_are_installed(self):
        self.assertEqual(self.used_model("mistral", "gemma3:1b", "gemma3:4b"), "gemma3:1b")

    def test_falls_back_to_first_model_when_no_gemma(self):
        self.assertEqual(self.used_model("llama3.2:latest", "mistral"), "llama3.2:latest")

    def test_explicit_model_overrides_gemma_priority(self):
        opener = make_opener(tags=tags_payload("gemma3:1b"), generate={"response": "ok"})
        model, _ = run_journal(opener, model="mistral")
        self.assertEqual(model, "mistral")
        self.assertEqual(opener.open.call_count, 1)  # never even asked which models exist


class FailureModes(NoRealNetwork):
    def assertJournalError(self, opener, pattern, **kw):
        with self.assertRaisesRegex(dc.OllamaError, pattern):
            run_journal(opener, **kw)

    def test_ollama_not_running(self):
        opener = make_opener(generate=urllib.error.URLError(ConnectionRefusedError("refused")))
        self.assertJournalError(opener, "Could not reach Ollama", model="gemma3:1b")

    def test_not_running_is_also_detected_when_listing_models(self):
        opener = make_opener(tags=urllib.error.URLError(ConnectionRefusedError("refused")))
        self.assertJournalError(opener, "Could not reach Ollama")

    def test_timeout_is_a_friendly_error(self):
        opener = make_opener(generate=TimeoutError("timed out"))
        self.assertJournalError(opener, "Could not reach Ollama", model="gemma3:1b")

    def test_unknown_model_404_suggests_pulling_it(self):
        opener = make_opener(generate=http_error(404))
        self.assertJournalError(opener, "ollama pull nope", model="nope")

    def test_server_error_500_has_no_pull_hint(self):
        opener = make_opener(generate=http_error(500))
        with self.assertRaises(dc.OllamaError) as ctx:
            run_journal(opener, model="gemma3:1b")
        self.assertIn("500", str(ctx.exception))
        self.assertNotIn("ollama pull", str(ctx.exception))

    def test_no_models_installed(self):
        opener = make_opener(tags=tags_payload())
        self.assertJournalError(opener, "no models")

    def test_empty_answer(self):
        opener = make_opener(generate={"response": "   "})
        self.assertJournalError(opener, "empty answer", model="gemma3:1b")

    def test_missing_response_field(self):
        opener = make_opener(generate={"done": True})
        self.assertJournalError(opener, "empty answer", model="gemma3:1b")

    def test_garbled_json(self):
        opener = make_opener(generate=b"<html>not json</html>")
        self.assertJournalError(opener, "could not read", model="gemma3:1b")


class StaysLocal(NoRealNetwork):
    def test_local_host_bypasses_system_proxy(self):
        opener = make_opener(generate={"response": "ok"})
        with mock.patch("urllib.request.build_opener", return_value=opener) as build:
            dc.ollama_journal(dets(), place=None, when=WHEN, model="gemma3:1b")
        handler = build.call_args.args[0]
        self.assertEqual(handler.proxies, {})  # ProxyHandler({}): never go through a proxy

    def test_remote_host_uses_default_opener(self):
        opener = make_opener(generate={"response": "ok"})
        with mock.patch("urllib.request.build_opener", return_value=opener) as build:
            dc.ollama_journal(dets(), place=None, when=WHEN, model="gemma3:1b", host="http://192.168.1.20:11434")
        self.assertEqual(build.call_args.args, ())

    def test_is_local(self):
        for host in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
            self.assertTrue(dc._is_local(host), host)
        for host in ("http://192.168.1.20:11434", "https://example.com", "http://localhost.evil.com"):
            self.assertFalse(dc._is_local(host), host)

    def test_networking_modules_are_not_imported_until_needed(self):
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import dawn_chorus; "
                "print(sorted(m for m in ('urllib.request', 'socket', 'requests', 'librosa') if m in sys.modules))"
                % str(Path(__file__).resolve().parent.parent))
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True).stdout.strip()
        self.assertNotIn("urllib.request", out)
        self.assertNotIn("requests", out)
        self.assertNotIn("librosa", out)


class PromptSafety(NoRealNetwork):
    def test_prompt_lists_only_detected_birds_and_forbids_invention(self):
        prompt = dc.journal_prompt(dets(), place="Test trail", when=WHEN)
        for needle in ("Black-capped Chickadee", "Dark-eyed Junco", "Test trail", "Tuesday, October 6, 2026",
                       "Use ONLY the facts below", "do not invent behaviour", "may contain errors"):
            self.assertIn(needle, prompt)
        self.assertNotIn("House Finch", prompt)

    def test_prompt_handles_missing_place(self):
        self.assertIn("Place: not given", dc.journal_prompt(dets(), place=None, when=WHEN))

    def test_model_text_is_escaped_in_the_notes_page(self):
        page = dc.render_notes(dets(), source="a.wav", duration=60, n_windows=20, when=WHEN, lat=None, lon=None,
                               place=None, lifers=set(), min_conf=0.5,
                               journal=("gemma3:1b", "<img src=x onerror=alert(1)>\n\nSecond paragraph."))
        self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;img", page)
        self.assertEqual(page.count("<p>"), 2)


if __name__ == "__main__":
    unittest.main()
