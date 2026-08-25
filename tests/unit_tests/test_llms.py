from typing import Any, Dict, Generator
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from langchain_aimlapi.llms import AimlapiLLM


@pytest.fixture(autouse=True)
def set_env_api_key(monkeypatch: MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("AIMLAPI_API_KEY", "testtoken")
    yield


def _dummy_response(payload: Dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = str(payload)
    return response


def test_call_routes_through_chat_completions(monkeypatch: MonkeyPatch) -> None:
    """AI/ML API has no /v1/completions endpoint (404); AimlapiLLM must send a
    single-turn chat-completions request instead and unwrap the message content.
    """
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], headers: Dict[str, str]) -> MagicMock:
        captured["url"] = url
        captured["json"] = json
        return _dummy_response({"choices": [{"message": {"content": "hello there"}}]})

    monkeypatch.setattr("langchain_aimlapi.llms.requests.post", fake_post)

    llm = AimlapiLLM(model="gpt-4o", max_tokens=50)
    result = llm.invoke("hi")

    assert result == "hello there"
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "prompt" not in captured["json"]
