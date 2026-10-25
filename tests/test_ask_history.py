import json

import pytest

from app import ask as ask_module
from app.ask import ask, build_chat_contents, load_history, save_history


def test_load_history_returns_empty_when_file_missing(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_history(missing) == []


def test_save_history_persists_messages(tmp_path):
    history_path = tmp_path / "chat.json"
    messages = [
        {"role": "user", "parts": [{"text": "こんにちは"}]},
        {"role": "model", "parts": [{"text": "こんにちは！"}]},
    ]

    save_history(history_path, messages)

    assert json.loads(history_path.read_text(encoding="utf-8")) == messages


def test_build_chat_contents_keeps_existing_history(tmp_path):
    history = [
        {"role": "user", "parts": [{"text": "前の質問"}]},
        {"role": "model", "parts": [{"text": "前の答え"}]},
    ]

    contents = build_chat_contents(history, "次の質問")

    assert contents == [
        {"role": "user", "parts": [{"text": "前の質問"}]},
        {"role": "model", "parts": [{"text": "前の答え"}]},
        {"role": "user", "parts": [{"text": "次の質問"}]},
    ]


def test_run_chat_supports_reset_and_exit(tmp_path, monkeypatch):
    history_path = tmp_path / "chat.json"
    save_history(history_path, [{"role": "user", "parts": [{"text": "古い質問"}]}])

    answers = iter(["/reset", "/exit"])
    lines = []

    monkeypatch.setattr(ask_module, "ask", lambda *args, **kwargs: "unused")
    ask_module.run_chat(history_path, input_func=lambda _: next(answers), output_func=lines.append)

    assert json.loads(history_path.read_text(encoding="utf-8")) == []
    assert any("初期化しました" in line for line in lines)


def test_ask_does_not_save_incomplete_history_on_api_error(tmp_path, monkeypatch):
    history_path = tmp_path / "chat.json"
    save_history(history_path, [{"role": "user", "parts": [{"text": "古い質問"}]}])

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        class models:
            @staticmethod
            def generate_content(*args, **kwargs):
                raise ask_module.errors.APIError(code=400, response_json={})

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)

    with pytest.raises(ask_module.errors.APIError):
        ask("新しい質問", history_path=history_path)

    assert json.loads(history_path.read_text(encoding="utf-8")) == [{"role": "user", "parts": [{"text": "古い質問"}]}]
