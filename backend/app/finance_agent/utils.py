"""Utility & helper functions."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
    return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """
    Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.

    """
    provider, model = fully_specified_name.split("/", maxsplit=1)

    model_parameters: dict[str, Any] = {
        "temperature": 0,
        "disable_streaming": False,
    }

    if "o1" in model:
        model_parameters["temperature"] = 1
        model_parameters["disable_streaming"] = True

    if "gpt-5" in model:
        model_parameters["reasoning_effort"] = "minimal"

    if "gpt-5.1" in model:
        model_parameters["reasoning_effort"] = "none"

    return init_chat_model(
        model=model,
        model_provider=provider,
        **model_parameters,
    )
