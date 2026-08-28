"""Test chat model integration."""

from typing import Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_aimlapi.chat_models import ChatAimlapi


class TestChatAimlapiUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatAimlapi]:
        return ChatAimlapi

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
        }


@tool
def _adder(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_bind_tools_uses_native_openai_implementation() -> None:
    """Regression test: ChatAimlapi must inherit BaseChatOpenAI's bind_tools.

    A stale `bind_tools = BaseChatModel.bind_tools` override on the class body
    shadowed the working, inherited BaseChatOpenAI implementation with the
    generic BaseChatModel stub, which unconditionally raises
    NotImplementedError. langchain-tests' own standard suite doesn't catch
    this: it self-disables (has_tool_calling probes bind_tools and silently
    treats a NotImplementedError as "no tool support", skipping the
    assertion instead of failing it) rather than failing loudly.
    """
    model = ChatAimlapi(model="bird-brain-001", api_key="dummytoken")

    assert model.bind_tools.__func__ is BaseChatOpenAI.bind_tools
    assert model.bind_tools.__func__ is not BaseChatModel.bind_tools
    assert (
        model.with_structured_output.__func__ is BaseChatOpenAI.with_structured_output
    )
    assert (
        model.with_structured_output.__func__
        is not BaseChatModel.with_structured_output
    )

    bound = model.bind_tools([_adder], tool_choice="any")
    assert isinstance(bound, RunnableBinding)
