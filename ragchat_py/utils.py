import re
import time
from typing import List, Union, Dict, Any, TypeVar
from types import BaseMessage, BaseLanguageModelInterface, OpenAIChatLanguageModel, ChatOptions, CustomPrompt
# TODO: check imports

# Utility function to sanitize the question
def sanitize_question(question: str) -> str:
    return re.sub(r"\n", " ", question.strip())

# Utility function to format facts (join strings by newlines)
def format_facts(facts: List[str]) -> str:
    return "\n".join(facts)

# Utility function to format chat history
def format_chat_history(chat_history: List[BaseMessage]) -> str:
    formatted_dialogue_turns = [
        f"Human: {turn.content}" if turn._get_type() == "human" else f"Assistant: {turn.content}"
        for turn in chat_history
    ]
    return format_facts(formatted_dialogue_turns)

# Define default chat options in a dictionary
DefaultChatOptions = {
    "streaming": True,
    "disable_rag": False,
    "session_id": "",
    "ratelimit_session_id": "",
    "similarity_threshold": 0.7,
    "top_k": 5,
    "history_length": 20,
    "history_ttl": 3600,
    "namespace": "default",
    "prompt_fn": None
}

T = TypeVar('T')
R = TypeVar('R')

# Utility to modify ChatOptions by overriding specific keys with defaults
def modify(obj: T, changes: Dict[str, Any]) -> T:
    result = {**obj, **changes}
    return result

# ModifiedChatOptions will be created from the original ChatOptions and DefaultChatOptions
ModifiedChatOptions = modify(ChatOptions, DefaultChatOptions)

# Default delay for timeouts
DEFAULT_DELAY = 20_000  # 20 seconds in milliseconds

# Delay function, which simulates async behavior
def delay(ms: int = DEFAULT_DELAY) -> None:
    time.sleep(ms / 1000)

# Type check function to identify if a model is OpenAIChatLanguageModel
def is_openai_chat_language_model(model: Union[BaseLanguageModelInterface, OpenAIChatLanguageModel]) -> bool:
    return hasattr(model, "specification_version")

