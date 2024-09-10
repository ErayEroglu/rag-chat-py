from typing import Optional, Dict, Union, Callable, Any
from langchain_core.messages.human import HumanMessage
from langchain.llms.base import BaseLanguageModel
from langchain.schema import BaseMessage
from langchain_core.language_models import BaseChatModel
from langsmith import traceable
from .types import SimpleNamespace, ChatOptions, UpstashMessage, OpenAIChatLanguageModel
from .utils import ModifiedChatOptions, is_openai_chat_language_model
#TODO: check modified object structure
from .logger import ChatLogger


class LLMService:
    def __init__(self, model: Union[BaseLanguageModel, OpenAIChatLanguageModel]):
        self.model = model

    async def call_llm(
        self,
        options_with_default: ModifiedChatOptions,
        _options: Optional[ChatOptions],
        callbacks: Dict[str, Optional[Callable[[str], None]]],
        debug: Optional[ChatLogger] = None
    ):
        return await traceable(
            lambda prompt: self._traceable_llm_call(prompt, options_with_default, callbacks, debug),
            {"name": "LLM Response", "metadata": {"sessionId": options_with_default.sessionId}}
        )

    async def _traceable_llm_call(
        self,
        prompt: str,
        options_with_default: ModifiedChatOptions,
        callbacks: Dict[str, Optional[Callable[[str], None]]],
        debug: Optional[ChatLogger]
    ):
        if debug:
            debug.start_llm_response()
        if options_with_default.streaming:
            return await self.make_streaming_llm_request(prompt, callbacks)
        else:
            return await self.make_llm_request(prompt, callbacks.get("onComplete"))

    async def make_streaming_llm_request(
        self,
        prompt: str,
        callbacks: Dict[str, Optional[Callable[[str], None]]]
    ):
        onComplete = callbacks.get("onComplete")
        onChunk = callbacks.get("onChunk")

        if is_openai_chat_language_model(self.model):

            #TODO: check if this part is working as expected
            # in the TS version, the model is invoked with a TS specific library
            stream = await self.model.stream([HumanMessage(prompt)])
        else:
            stream = await self.model.stream([HumanMessage(prompt)])

        concatenated_output = ""

        async for value in stream:
            if isinstance(value, str):
                yield value
            else:
                message = value.content or ""
                if onChunk:
                    onChunk({
                        "content": message,
                        "inputTokens": value.usage_metadata.input_tokens or 0,
                        "chunkTokens": value.usage_metadata.output_tokens or 0,
                        "totalTokens": value.usage_metadata.total_tokens or 0,
                        "rawContent": value
                    })
                concatenated_output += message
                yield message

        if onComplete:
            onComplete(concatenated_output)

    async def make_llm_request(self, prompt: str, onComplete: Optional[Callable[[str], None]]):
        if is_openai_chat_language_model(self.model):
            # Using LangChain's `invoke` method instead of `generateText`
            response = await self.model.invoke(HumanMessage(prompt))
            content = response.content
        else:
            response = await self.model.invoke(HumanMessage(prompt))
            content = response.content

        if onComplete:
            onComplete(content)
        return {"output": content, "isStream": False}
