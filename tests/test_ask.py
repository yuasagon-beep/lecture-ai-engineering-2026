from types import SimpleNamespace

from app.ask import format_usage_metadata


def test_format_usage_metadata_includes_prompt_and_response_counts():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=123,
            response_token_count=45,
            total_token_count=168,
        )
    )

    text = format_usage_metadata(response)

    assert "入力トークン数: 123" in text
    assert "出力トークン数: 45" in text
    assert "合計トークン数: 168" in text


def test_format_usage_metadata_handles_missing_usage_metadata():
    response = SimpleNamespace()

    assert format_usage_metadata(response) == "トークン使用量: 取得できませんでした"
