from typing import Optional, Dict, Union, Callable, Any
from langchain.core.messages import HumanMessage
from langchain.core.language_models.base import BaseLanguageModelInterface
from langchain.core.utils.stream import IterableReadableStreamInterface
from langchain.core.logger import ChatLogger
from langchain.core.utils import stream_text, generate_text
from langsmith.traceable import traceable
from types import SimpleNamespace
from utils import ModifiedChatOptions, ChatOptions


class LLMService:
    def __init__(self, model: Union[BaseLanguageModelInterface, Any]):
        self.model = model

    def call_llm(self, options_with_default: ModifiedChatOptions, _options: Optional[ChatOptions], callbacks: Dict[str, Optional[Callable]], debug: Optional[ChatLogger] = None):
        return traceable(
            lambda prompt: self._traceable_llm_call(prompt, options_with_default, callbacks, debug),
            {"name": "LLM Response", "metadata": {"sessionId": options_with_default.sessionId}}
        )

    def _traceable_llm_call(self, prompt: str, options_with_default: ModifiedChatOptions, callbacks: Dict[str, Optional[Callable]], debug: Optional[ChatLogger]):
        debug and debug.start_llm_response()
        if options_with_default.streaming:  
            return self.make_streaming_llm_request(prompt, callbacks)
        else:
            return self.make_llm_request(prompt, callbacks.get("onComplete"))

    async def make_streaming_llm_request(self, prompt: str, callbacks: Dict[str, Optional[Callable]]):
        onComplete = callbacks.get("onComplete")
        onChunk = callbacks.get("onChunk")

        if isinstance(self.model, BaseLanguageModelInterface):
            text_stream = await stream_text(model=self.model, prompt=prompt)
            stream = text_stream["textStream"]
        else:
            stream = await self.model.stream([HumanMessage(prompt)])

        reader = stream.get_reader()
        concatenated_output = ""

        async def process_stream(controller):
            try:
                while True:
                    done, value = await reader.read()
                    if done:
                        break
                    if isinstance(value, str):
                        controller.enqueue(value)
                        continue
                    else:
                        message = value.content or ""
                        onChunk and onChunk({
                            "content": message,
                            "inputTokens": value.usage_metadata.input_tokens or 0,
                            "chunkTokens": value.usage_metadata.output_tokens or 0,
                            "totalTokens": value.usage_metadata.total_tokens or 0,
                            "rawContent": value
                        })
                        concatenated_output += message
                        controller.enqueue(message)
                controller.close()
                onComplete and onComplete(concatenated_output)
            except Exception as error:
                controller.error(error)

        new_stream = SimpleNamespace(start=lambda controller: process_stream(controller))
        return {"output": new_stream, "isStream": True}

    async def make_llm_request(self, prompt: str, onComplete: Optional[Callable[[str], None]]):
        if isinstance(self.model, BaseLanguageModelInterface):
            text = await generate_text(model=self.model, prompt=prompt)
            content = text["text"]
        else:
            text = await self.model.invoke(prompt)
            content = text.content
        onComplete and onComplete(content)
        return {"output": content, "isStream": False}