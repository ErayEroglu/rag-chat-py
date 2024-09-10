from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Tuple
from .error.model import UpstashError
from langsmith import traceable
from .config import Config
from .constants import (
    DEFAULT_CHAT_RATELIMIT_SESSION_ID,
    DEFAULT_CHAT_SESSION_ID,
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_HISTORY_TTL,
    DEFAULT_NAMESPACE,
    DEFAULT_PROMPT_WITHOUT_RAG,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)
from .context_service.index import ContextService
from .database import Database
from .error.ratelimit import RatelimitUpstashError
from .error.vector import UpstashVectorError
from .history_service.index import HistoryService
from .history_service.in_memory_history import InMemoryHistory
from .history_service.redis_custom_history import UpstashRedisHistory
from .llm_service import LLMService
from .logger import ChatLogger
from .ratelimit_service import RateLimitService
from .types import ChatOptions, RAGChatConfig
from .utils import sanitize_question, ModifiedChatOptions

TMetadata = TypeVar('TMetadata', bound=Dict[str, Any])
TChatOptions = TypeVar('TChatOptions', bound=ChatOptions)


class RAGChat:
    def __init__(self, config: Optional[RAGChatConfig] = None):
        self.config = Config(config)

        if not self.config.vector:
            raise UpstashVectorError("Vector can not be undefined!")

        if not self.config.model:
            raise UpstashError("Model can not be undefined!")

        vector_service = Database(self.config.vector)
        self.history = HistoryService(redis=self.config.redis).service
        self.llm = LLMService(self.config.model)
        self.context = ContextService(vector_service, self.config.namespace or DEFAULT_NAMESPACE)
        self.debug = ChatLogger(log_level="INFO", log_output="console") if self.config.debug else None
        self.ratelimit = RateLimitService(self.config.ratelimit)

    def chat(self, input: str, options: Optional[ChatOptions] = None) -> Any:
        @traceable
        async def inner_chat(input: str, options: Optional[ChatOptions] = None) -> Any:
            try:
                options_with_default = self.get_options_with_defaults(options)
                # Checks ratelimit of the user. If not enabled `success` will be always true.
                await self.check_ratelimit(options_with_default)

                # Add the user message to chat history
                await self.add_user_message_to_history(input, options_with_default)

                # Sanitize the input by stripping newline chars.
                question = sanitize_question(input)
                context, metadata = await self.context._get_context(
                    options_with_default, input, self.debug
                )(options_with_default.session_id)

                formatted_history = await self.get_chat_history(options_with_default)

                prompt = await self.generate_prompt(
                    options_with_default, context, question, formatted_history
                )

                # Call the LLM service
                llm_result = await self.llm.call_llm(
                    options_with_default, options, {
                        'onChunk': options_with_default.on_chunk,
                        'onComplete': self.handle_completion,
                    }, self.debug
                )(prompt)

                return {
                    **llm_result,
                    'metadata': metadata,
                }

            except Exception as error:
                if self.debug:
                    await self.debug.log_error(error)
                raise error 

        # Setting up tracing configuration
        tracing_config = {
            'name': "Rag Chat",
            'tracingEnabled': bool(globals().get('globalTracer')),
            'client': globals().get('globalTracer', None),
            'project_name': "Upstash Rag Chat",
            'tags': ["streaming" if options and options.streaming else "non-streaming"],
            'metadata': self.get_options_with_defaults(options),
        }

        # Call the traceable function
        return traceable(inner_chat, tracing_config)(input, options)

    async def generate_prompt(
        self,
        options_with_default: ModifiedChatOptions,
        context: str,
        question: str,
        formatted_history: str
    ) -> str:
        async def inner_generate_prompt(
            options_with_default: ModifiedChatOptions,
            context: str,
            question: str,
            formatted_history: str
        ) -> str:
            prompt = options_with_default.prompt_fn({
                'context': context,
                'question': question,
                'chatHistory': formatted_history,
            })
            await self.debug.log_final_prompt(prompt)
            return prompt

        return traceable(inner_generate_prompt, {
            'name': "Final Prompt",
            'run_type': "prompt"
        })(options_with_default, context, question, formatted_history)

    async def get_chat_history(self, options_with_default: ModifiedChatOptions) -> str:
        async def inner_get_chat_history(options_with_default: ModifiedChatOptions) -> str:
            self.debug.start_retrieve_history()
            original_chat_history = await self.history.get_messages({
                'session_id': options_with_default.session_id,
                'amount': options_with_default.history_length,
            })

            cloned_chat_history = original_chat_history[:]  # Deep copy of the list
            modified_chat_history = (
                await options_with_default.on_chat_history_fetched(cloned_chat_history)
            ) or original_chat_history

            self.debug.end_retrieve_history(cloned_chat_history)

            formatted_history = "\n".join(
                f"USER MESSAGE: {m['content']}" if m['role'] == "user"
                else f"YOUR MESSAGE: {m['content']}"
                for m in reversed(modified_chat_history)
            )

            await self.debug.log_retrieve_format_history(formatted_history)
            return formatted_history

        return traceable(inner_get_chat_history, {
            'name': "Retrieve History",
            'tags': [self.config.redis if self.config.redis else "in-memory"],
            'metadata': {'session_id': options_with_default.session_id},
            'run_type': "retriever"
        })(options_with_default)

    async def add_user_message_to_history(
        self,
        input: str,
        options_with_default: ModifiedChatOptions
    ) -> None:
        await self.history.add_message({
            'message': {'content': input, 'role': "user"},
            'session_id': options_with_default.session_id,
        })

    async def check_ratelimit(self, options_with_default: ModifiedChatOptions) -> None:
        ratelimit_response = await self.ratelimit.check_limit(
            options_with_default.ratelimit_session_id
        )

        options_with_default.ratelimit_details(ratelimit_response)
        if not ratelimit_response.success:
            raise RatelimitUpstashError("Couldn't process chat due to ratelimit.", {
                'error': "ERR:USER_RATELIMITED",
                'resetTime': ratelimit_response.reset,
            })

    def get_options_with_defaults(self, options: Optional[ChatOptions] = None) -> ModifiedChatOptions:
        is_rag_disabled_and_prompt_function_missing = options.disable_rag and not options.prompt_fn if options else False

        return ModifiedChatOptions(
            on_chat_history_fetched=options.on_chat_history_fetched if options else None,
            on_context_fetched=options.on_context_fetched if options else None,
            on_chunk=options.on_chunk if options else None,
            ratelimit_details=options.ratelimit_details if options else None,
            metadata=options.metadata if options else self.config.metadata,
            namespace=options.namespace if options else self.config.namespace or DEFAULT_NAMESPACE,
            streaming=options.streaming if options else self.config.streaming or False,
            session_id=options.session_id if options else self.config.session_id or DEFAULT_CHAT_SESSION_ID,
            disable_rag=options.disable_rag if options else False,
            similarity_threshold=options.similarity_threshold if options else DEFAULT_SIMILARITY_THRESHOLD,
            top_k=options.top_k if options else DEFAULT_TOP_K,
            history_length=options.history_length if options else DEFAULT_HISTORY_LENGTH,
            history_ttl=options.history_ttl if options else DEFAULT_HISTORY_TTL,
            ratelimit_session_id=options.ratelimit_session_id if options else self.config.ratelimit_session_id or DEFAULT_CHAT_RATELIMIT_SESSION_ID,
            prompt_fn=DEFAULT_PROMPT_WITHOUT_RAG if is_rag_disabled_and_prompt_function_missing else (options.prompt_fn if options else self.config.prompt)
        )
