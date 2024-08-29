from typing import Union, List, Dict, Any, Optional, Callable
from langsmith.traceable import traceable
from nanoid import generate as nanoid
from database import Database, AddContextPayload, ResetOptions, VectorPayload
from logger import ChatLogger
from utils import format_facts, ModifiedChatOptions

class ContextService:
    def __init__(self, vector_service: Database, namespace: str):
        self._vector_service = vector_service
        self.namespace = namespace

    async def add(self, args: Union[AddContextPayload, str]):
        if isinstance(args, str):
            result = await self._vector_service.save({
                "type": "text",
                "data": args,
                "id": nanoid(),
                "options": {"namespace": self.namespace},
            })
            return result
        return await self._vector_service.save(args)

    async def add_many(self, args: Union[List[AddContextPayload], List[str]]):
        return [await self.add(data) for data in args]

    async def delete_entire_context(self, options: Optional[ResetOptions] = None):
        await self._vector_service.reset(
            {"namespace": options.namespace} if options and options.namespace else None
        )

    async def delete(self, id: Union[str, List[str]], namespace: Optional[str] = None):
        await self._vector_service.delete({"ids": [id] if isinstance(id, str) else id, "namespace": namespace})

    # TODO: Check if this function works as expected
    # Not so sure about its functionality
    def _get_context(self, options_with_default: ModifiedChatOptions, input: str, debug: Optional[ChatLogger] = None):
        async def retrieve_context(session_id: str):
            await debug.log_send_prompt(input) if debug else None
            debug.start_retrieve_context() if debug else None

            if options_with_default.disable_rag:
                return {"formatted_context": "", "metadata": []}

            async def fetch_context(payload: VectorPayload):
                original_context = await self._vector_service.retrieve(payload)
                cloned_context = original_context.copy()
                return await options_with_default.on_context_fetched(cloned_context) if options_with_default.on_context_fetched else original_context

            context = await traceable(
                fetch_context,
                {"name": "Step: Fetch", "metadata": {"session_id": session_id}, "run_type": "retriever"}
            )({
                "question": input,
                "similarity_threshold": options_with_default.similarity_threshold,
                "top_k": options_with_default.top_k,
                "namespace": options_with_default.namespace,
            })

            await debug.end_retrieve_context(context) if debug else None

            formatted_context = await traceable(
                lambda _context: format_facts([data["data"] for data in _context]),
                {"name": "Step: Format", "metadata": {"session_id": session_id}, "run_type": "tool"}
            )(context)

            return {
                "formatted_context": formatted_context,
                "metadata": [data["metadata"] for data in context]
            }

        return traceable(
            retrieve_context,
            {"name": "Retrieve Context", "metadata": {"session_id": options_with_default.session_id, "namespace": options_with_default.namespace}}
        )