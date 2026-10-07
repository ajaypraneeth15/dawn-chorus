"""Tests for the optional Gemma walk companion (`companion` command, `ollama_chat`).

Run with:  python3 -m unittest discover -s tests -v

Nothing here needs a real Ollama server, GPU, or model download. Two fakes are used:
  * unittest.mock replaces urllib's opener (a guard fails any real socket connection), and
  * the FakeOllama stand-in server from test_dawn_chorus.py, for an end-to-end run of the command.
These prove our client code and our conversation logic. They say nothing about how good the
sentences from a real Gemma model are.
"""
import contextlib
import datetime as dt
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dawn_chorus as dc  # noqa: E402
from test_dawn_chorus import MODELS, FakeOllama  # noqa: E402
from test_ollama_mock import (NoRealNetwork, FakeResponse, http_error, make_opener,  # noqa: E402
                              tags_payload)

WHEN = dt.date(2026, 10, 7)


def chat_reply(text):
    return {"model": "gemma3:1b", "done": True, "message": {"role": "assistant", "content": text}}


def card():
    sp = lambda sci, common: dc.Species(sci, common)  # noqa: E731
    return [(sp("Poecile atricapillus", "Black-capped Chickadee"), 0.8),
            (sp("Junco hyemalis", "Dark-eyed Junco"), 0.3),
            (sp("Setophaga striata", "Blackpoll Warbler"), 0.1)]


def run_chat(opener, messages=None, **kw):
    messages = messages or [{"role": "user", "content": "hi"}]
    with mock.patch("urllib.request.build_opener", return_value=opener):
        return dc.ollama_chat(messages, **kw)


class OllamaChatClient(NoRealNetwork):
    def test_posts_history_to_api_chat(self):
        opener = make_opener(generate=chat_reply("  Let's go.\n"))
        history = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
        model, text = run_chat(opener, history, model="gemma3:1b")
        self.assertEqual((model, text), ("gemma3:1b", "Let's go."))
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(req.data)
        self.assertEqual(body["messages"], history)
        self.assertEqual(body["model"], "gemma3:1b")
        self.assertIs(body["stream"], False)

    def test_prefers_gemma_when_no_model_is_named(self):
        opener = make_opener(tags=tags_payload("llama3.2:latest", "gemma3:4b"), generate=chat_reply("ok"))
        model, _ = run_chat(opener)
        self.assertEqual(model, "gemma3:4b")
        self.assertEqual(json.loads(opener.open.call_args.args[0].data)["model"], "gemma3:4b")

    def test_explicit_model_skips_the_lookup(self):
        opener = make_opener(tags=tags_payload("gemma3:1b"), generate=chat_reply("ok"))
        self.assertEqual(run_chat(opener, model="mistral")[0], "mistral")
        self.assertEqual(opener.open.call_count, 1)

    def assertChatError(self, opener, pattern, **kw):
        with self.assertRaisesRegex(dc.OllamaError, pattern):
            run_chat(opener, **kw)

    def test_ollama_not_running(self):
        self.assertChatError(make_opener(generate=urllib.error.URLError(ConnectionRefusedError("refused"))),
                             "Could not reach Ollama", model="gemma3:1b")

    def test_unknown_model_404_suggests_pull(self):
        self.assertChatError(make_opener(generate=http_error(404)), "ollama pull nope", model="nope")

    def test_auto_picked_model_is_named_in_the_404_hint(self):
        self.assertChatError(make_opener(tags=tags_payload("gemma3:1b"), generate=http_error(404)),
                             "ollama pull gemma3:1b")

    def test_empty_missing_and_odd_replies(self):
        self.assertChatError(make_opener(generate=chat_reply("  ")), "empty answer", model="m")
        self.assertChatError(make_opener(generate={"done": True}), "empty answer", model="m")
        self.assertChatError(make_opener(generate={"message": None}), "empty answer", model="m")
        self.assertChatError(make_opener(generate=["not", "a", "dict"]), "could not read", model="m")
        self.assertChatError(make_opener(generate=b"<html>nope</html>"), "could not read", model="m")


class SystemPrompt(unittest.TestCase):
    def prompt(self, **kw):
        kw.setdefault("place", "Cayuga trail")
        kw.setdefault("life_count", 7)
        kw.setdefault("recent", ["House Finch", "Blue Jay"])
        return dc.companion_system_prompt(card(), when=WHEN, **kw)

    def test_contains_only_the_real_facts(self):
        text = self.prompt()
        for needle in ("Black-capped Chickadee: likely this week", "Dark-eyed Junco: possible this week",
                       "Blackpoll Warbler: lucky find this week", "Wednesday, October 7, 2026", "week 37 of 48",
                       "Cayuga trail", "Life list: 7 species", "House Finch, Blue Jay"):
            self.assertIn(needle, text)
        self.assertNotIn("Northern Cardinal", text)

    def test_job_is_to_send_the_person_outside_and_not_invent(self):
        text = self.prompt()
        for needle in ("OUTSIDE", "at most 3 short sentences", "ONLY the birds in the bingo card",
                       "Do not describe their songs", "Do not invent weather"):
            self.assertIn(needle, text)

    def test_missing_place_recent_and_language(self):
        text = self.prompt(place=None, life_count=0, recent=[])
        self.assertIn("Place: not given", text)
        self.assertIn("none yet", text)
        self.assertFalse(any(line.startswith("Reply in") for line in text.splitlines()))
        self.assertIn("Reply in Tamil.", self.prompt(language="Tamil"))


class Session(unittest.TestCase):
    """The conversation loop, with scripted inputs and a fake `chat` function."""

    def run_session(self, inputs=(), *, turns=4, once=False, chat=None):
        said, asked = [], []
        feed = iter(inputs)

        def ask(prompt):
            asked.append(prompt)
            value = next(feed)
            if isinstance(value, BaseException):
                raise value
            return value

        calls = []

        def default_chat(messages):
            calls.append([dict(m) for m in messages])
            return f"reply {len(calls)}"

        history = dc.companion_session(chat or default_chat, "SYSTEM", turns=turns, once=once, ask=ask, say=said.append)
        return history, said, asked, calls

    def test_opens_with_a_mission_then_answers_each_message(self):
        history, said, asked, calls = self.run_session(["what should I wear?", "go"])
        self.assertEqual(said, ["reply 1", "reply 2"])
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(history[0]["content"], "SYSTEM")
        self.assertEqual(history[1]["content"], dc.MISSION_REQUEST)
        self.assertEqual(history[3]["content"], "what should I wear?")
        self.assertEqual(len(calls[1]), 4)  # the model sees the whole history so far

    def test_turn_cap_ends_the_chat_even_if_the_person_keeps_typing(self):
        history, said, asked, calls = self.run_session(["more"] * 50, turns=3)
        self.assertEqual(len(calls), 4)  # 1 mission + 3 replies
        self.assertEqual(len(asked), 3)

    def test_zero_turns_and_once_never_ask(self):
        for kwargs in ({"turns": 0}, {"once": True}):
            _, said, asked, calls = self.run_session(["unused"], **kwargs)
            self.assertEqual((len(said), asked), (1, []), kwargs)

    def test_empty_line_or_bye_ends_immediately(self):
        for word in ("", "  ", "go", "BYE", "Quit"):
            _, said, _, calls = self.run_session([word])
            self.assertEqual(len(calls), 1, repr(word))

    def test_eof_and_ctrl_c_end_quietly(self):
        for exc in (EOFError(), KeyboardInterrupt()):
            _, said, _, _ = self.run_session([exc])
            self.assertEqual(said, ["reply 1"])

    def test_losing_ollama_mid_chat_is_handled(self):
        count = {"n": 0}

        def flaky(messages):
            count["n"] += 1
            if count["n"] == 2:
                raise dc.OllamaError("Could not reach Ollama at x.")
            return "ok"

        history, said, _, _ = self.run_session(["hi", "again"], chat=flaky)
        self.assertIn("lost its connection", said[-1])
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant"])  # failed turn removed

    def test_failure_on_the_very_first_turn_propagates(self):
        def down(messages):
            raise dc.OllamaError("Could not reach Ollama at x.")

        with self.assertRaises(dc.OllamaError):
            self.run_session([], chat=down)


@unittest.skipUnless(MODELS, "BirdNET weights not installed")
class CompanionCommand(unittest.TestCase):
    """`python dawn_chorus.py companion ...` end to end against the stand-in Ollama server."""

    def run_cmd(self, *extra, inputs=None, host):
        tmp = tempfile.mkdtemp()
        argv = ["--db", str(Path(tmp) / "life.db"), "companion", "--lat", "42.45", "--lon", "-76.5",
                "--place", "Test trail", "--date", "2026-10-07", "--ollama-host", host, *extra]
        out = io.StringIO()
        feed = mock.patch("builtins.input", side_effect=inputs) if inputs is not None else contextlib.nullcontext()
        with contextlib.redirect_stdout(out), feed:
            dc.main(argv)
        return out.getvalue()

    def test_once_prints_the_mission_and_sends_you_outside(self):
        with FakeOllama(reply="Try for three birds on your card, then pocket the phone.") as fake:
            out = self.run_cmd("--once", host=fake.url)
        self.assertIn("Try for three birds on your card", out)
        self.assertIn("Time to go outside", out)
        self.assertIn("listen <your recording> --lat 42.45 --lon -76.5 --journal", out)
        self.assertEqual(fake.paths[-1], "/api/chat")
        sent = fake.requests[-1]
        self.assertEqual(sent["model"], "gemma3:1b")  # Gemma preferred over llama3.2
        system = sent["messages"][0]["content"]
        import re
        card_lines = [line for line in system.splitlines()
                      if re.fullmatch(r"- [^:]+: (likely|possible|lucky find) this week", line)]
        self.assertEqual(len(card_lines), 24)  # the whole bingo card, and nothing else
        self.assertIn("Test trail", system)

    def test_interactive_chat_is_capped_and_sends_history(self):
        with FakeOllama(reply="Sounds good.") as fake:
            out = self.run_cmd("--turns", "2", inputs=["one", "two", "three", "four"], host=fake.url)
        self.assertEqual(len(fake.requests), 3)  # mission + 2 answers, then the cap
        self.assertEqual([m["content"] for m in fake.requests[-1]["messages"] if m["role"] == "user"][1:], ["one", "two"])
        self.assertIn("Sounds good.", out)

    def test_language_option_reaches_the_model(self):
        with FakeOllama() as fake:
            self.run_cmd("--once", "--language", "Tamil", host=fake.url)
        self.assertIn("Reply in Tamil.", fake.requests[-1]["messages"][0]["content"])

    def test_ollama_not_running_gives_a_friendly_exit_and_points_to_plan(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cmd("--once", host="http://127.0.0.1:9")  # nothing listens on port 9
        msg = str(ctx.exception)
        self.assertIn("companion unavailable", msg)
        self.assertIn("Could not reach Ollama", msg)
        self.assertIn("plan --lat 42.45 --lon -76.5", msg)

    def test_remote_host_is_flagged_before_anything_is_sent(self):
        out = io.StringIO()
        with mock.patch.object(dc, "ollama_chat", side_effect=dc.OllamaError("offline")) as chat, \
                contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            dc.main(["--db", str(Path(tempfile.mkdtemp()) / "l.db"), "companion", "--lat", "42.45", "--lon", "-76.5",
                     "--once", "--ollama-host", "http://192.168.1.20:11434"])
        self.assertIn("is not this machine", out.getvalue())
        self.assertEqual(chat.call_args.kwargs["host"], "http://192.168.1.20:11434")


if __name__ == "__main__":
    unittest.main()
