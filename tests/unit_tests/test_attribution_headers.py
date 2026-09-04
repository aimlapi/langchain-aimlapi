"""Attribution headers must survive a caller-supplied ``default_headers``.

``default_headers`` carries the four headers AI/ML API uses to attribute a call
to the LangChain integration. It is a pydantic field with a ``default_factory``,
and a ``default_factory`` fires only when the field is *not supplied* -- so
before the merge helper existed, a caller who passed a single header of their
own replaced the dict wholesale and sent none of ours. The request still
succeeded, which is what made it expensive: the spend simply went untracked and
neither side could see it happen.
"""

import pytest

from langchain_aimlapi import ChatAimlapi
from langchain_aimlapi.constants import AIMLAPI_HEADERS, merge_aimlapi_headers
from langchain_aimlapi.embeddings import AimlapiEmbeddings

ATTRIBUTION_KEYS = {
    "HTTP-Referer",
    "X-Title",
    "X-AIMLAPI-Partner-ID",
    "X-AIMLAPI-Source",
}


def _chat(**kwargs):
    return ChatAimlapi(model="openai/gpt-4o-mini", api_key="dummy", **kwargs)


def _embeddings(**kwargs):
    return AimlapiEmbeddings(
        model="openai/text-embedding-3-small", api_key="dummy", **kwargs
    )


@pytest.mark.parametrize("build", [_chat, _embeddings], ids=["chat", "embeddings"])
def test_attribution_present_when_no_headers_supplied(build):
    assert ATTRIBUTION_KEYS <= set(build().default_headers)


@pytest.mark.parametrize("build", [_chat, _embeddings], ids=["chat", "embeddings"])
def test_caller_header_does_not_drop_attribution(build):
    """The regression this file exists for."""
    headers = build(default_headers={"X-App": "my-app"}).default_headers
    assert ATTRIBUTION_KEYS <= set(headers)
    assert headers["X-App"] == "my-app"


@pytest.mark.parametrize("build", [_chat, _embeddings], ids=["chat", "embeddings"])
def test_naming_one_of_our_keys_still_wins(build):
    """Overriding by name stays possible -- a reseller may need it."""
    headers = build(default_headers={"X-Title": "Downstream"}).default_headers
    assert headers["X-Title"] == "Downstream"
    assert headers["X-AIMLAPI-Partner-ID"] == AIMLAPI_HEADERS["X-AIMLAPI-Partner-ID"]


def test_embeddings_none_means_not_supplied():
    """``None`` is a declared value for this field on the embeddings model."""
    assert ATTRIBUTION_KEYS <= set(_embeddings(default_headers=None).default_headers)


@pytest.mark.parametrize("build", [_chat, _embeddings], ids=["chat", "embeddings"])
def test_instances_do_not_share_mutable_state_with_the_constant(build):
    headers = build().default_headers
    headers["X-Scratch"] = "1"
    assert "X-Scratch" not in AIMLAPI_HEADERS
    assert "X-Scratch" not in build().default_headers


def test_merge_helper_returns_a_fresh_dict():
    assert merge_aimlapi_headers() == AIMLAPI_HEADERS
    assert merge_aimlapi_headers() is not AIMLAPI_HEADERS
