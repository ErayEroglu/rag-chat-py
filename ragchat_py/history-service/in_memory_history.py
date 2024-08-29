from ..constants import DEFAULT_CHAT_SESSION_ID, DEFAULT_HISTORY_LENGTH
from .base_message_history import BaseMessageHistory, HistoryAddMessage, Message, UpstashMessage
from typing import List, Dict

# Global store to simulate in-memory storage
global_store : Dict[str, List[Message]]  = {}

# InMemoryHistory class implementing BaseMessageHistory
class InMemoryHistory(BaseMessageHistory):
    def __init__(self):
        if not global_store:
            global_store.clear()

    async def add_message(self, data: HistoryAddMessage) -> None:
        session_id = data.get("sessionId", DEFAULT_CHAT_SESSION_ID)
        if session_id not in global_store:
            global_store[session_id] = {"messages": []}

        old_messages = global_store[session_id]["messages"]
        new_messages = [
            {
                **data["message"],
                # "__internal_order": len(old_messages),
            },
            *old_messages,
        ]

        global_store[session_id]["messages"] = new_messages

    async def delete_messages(self, session_id: str) -> None:
        if session_id not in global_store:
            return

        global_store[session_id]["messages"] = []

    async def get_messages(self, session_id: str = DEFAULT_CHAT_SESSION_ID, amount: int = DEFAULT_HISTORY_LENGTH) -> List[UpstashMessage]:
        if session_id not in global_store:
            global_store[session_id] = {"messages": []}

        messages = global_store.get(session_id, {}).get("messages", [])
        sorted_messages = messages[:amount][::-1]
        messages_with_id = [{**message, "id": str(index)} for index, message in enumerate(sorted_messages)]

        return messages_with_id