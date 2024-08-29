from typing import List, Dict, Any, Optional, TypedDict
from abc import ABC, abstractmethod
from ..types import UpstashMessage

class Message:
    def __init__(self,
                 role: str,
                 content: str,
                 metadata: Optional[Dict] = None,
                 usage_metadata: Optional[Dict] = None
    ):
        self.role = role
        self.content = content
        self.metadata = metadata
        self.usage_metadata = usage_metadata

class HistoryAddMessage(TypedDict):
    message: Message
    sessionId: Optional[str]
    sessionTTL: Optional[int]

class BaseMessageHistory(ABC):
    @abstractmethod
    async def get_messages(self, session_id: str, amount: int) -> List[UpstashMessage]:
        pass

    @abstractmethod
    async def add_message(self, data: HistoryAddMessage) -> None:
        pass

    @abstractmethod
    async def delete_messages(self, session_id: str) -> None:
        pass