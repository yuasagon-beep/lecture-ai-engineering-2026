"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
import os
from pathlib import Path

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / ".gemini_chat_history.json"


def get_history_path(path: str | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env_path = os.getenv("AIE_CHAT_HISTORY_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_HISTORY_PATH


def load_history(path: str | Path | None = None) -> list[dict]:
    history_path = Path(path) if path is not None else get_history_path()
    if not history_path.exists():
        return []

    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []
    return data


def save_history(path_or_history, history=None) -> None:
    if isinstance(path_or_history, (str, Path)):
        if history is None:
            raise TypeError("save_history requires history data and a target path")
        history_path = Path(path_or_history)
        history_data = history
    else:
        history_data = path_or_history
        if isinstance(history, (str, Path)):
            history_path = Path(history)
        else:
            history_path = get_history_path()

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_chat_contents(history: list[dict], question: str) -> list[dict]:
    return [
        *history,
        {"role": "user", "parts": [{"text": question}]},
    ]


def format_usage_metadata(response) -> str:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return "トークン使用量: 取得できませんでした"

    prompt_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "response_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    pieces = []
    if prompt_tokens is not None:
        pieces.append(f"入力トークン数: {prompt_tokens}")
    if output_tokens is not None:
        pieces.append(f"出力トークン数: {output_tokens}")
    if total_tokens is not None:
        pieces.append(f"合計トークン数: {total_tokens}")

    return " | ".join(pieces) if pieces else "トークン使用量: 取得できませんでした"


def ask(question: str, history_path: str | Path | None = None, reset_history: bool = False) -> str:
    resolved_path = Path(history_path) if history_path is not None else get_history_path()
    if reset_history and resolved_path.exists():
        resolved_path.unlink()

    history = load_history(resolved_path)
    contents = build_chat_contents(history, question)

    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=contents)
    answer = response.text

    updated_history = [
        *history,
        {"role": "user", "parts": [{"text": question}]},
        {"role": "model", "parts": [{"text": answer}]},
    ]
    save_history(updated_history, resolved_path)

    print(f"[トークン使用量] {format_usage_metadata(response)}")
    return answer


def run_chat(history_path: str | Path | None = None, *, input_func=input, output_func=print, reset: bool = False) -> None:
    resolved_path = Path(history_path) if history_path is not None else get_history_path()
    if reset:
        save_history([], resolved_path)
        history = []
    else:
        history = load_history(resolved_path)

    output_func(f"[chat] 対話モードで会話を続けます（/reset で初期化 /exit で終了）")
    output_func(f"[history] {resolved_path}")

    while True:
        try:
            message = input_func("\n> ")
        except EOFError:
            output_func("")
            return

        command = message.strip()
        if command == "":
            continue
        if command == "/reset":
            history = []
            save_history(history, resolved_path)
            output_func("会話履歴を初期化しました。新しい会話を開始します。")
            continue
        if command == "/exit":
            save_history(history, resolved_path)
            output_func("対話モードを終了しました。履歴は保存されています。")
            return

        try:
            answer = ask(command, history_path=resolved_path)
        except errors.APIError as e:
            output_func(api_error_message(e.code, e.message))
            continue

        history = load_history(resolved_path)
        output_func(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保存しながら Gemini に質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--history-file", default=None, help="会話履歴の保存先 JSON ファイル")
    parser.add_argument("--reset", action="store_true", help="履歴をクリアして新しい会話を開始")
    parser.add_argument("--chat", action="store_true", help="対話モード（REPL）で連続質問を行う")
    args = parser.parse_args()

    history_path = get_history_path(args.history_file)

    if args.chat:
        run_chat(history_path, reset=args.reset)
        return

    print(f"[model] {MODEL}")
    print(f"[history] {history_path}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, history_path=history_path, reset_history=args.reset))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
