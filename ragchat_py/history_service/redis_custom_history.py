from typing import Optional, Dict, Any, List, TypedDict
from redis import Redis
from .base_message_history import BaseMessageHistory, HistoryAddMessage
from ..constants import DEFAULT_CHAT_SESSION_ID, DEFAULT_HISTORY_LENGTH
from ..types import UpstashMessage

# check if this structure is correct
class UpstashRedisHistoryConfig(TypedDict):
    config: Optional[Dict[str, Any]]
    client: Optional[Redis]

# Define the UpstashRedisHistory class
class UpstashRedisHistory(BaseMessageHistory):
    def __init__(self, _config: UpstashRedisHistoryConfig):
        config = _config.get('config')
        client = _config.get('client')

        if client:
            self.client = client
        elif config:
            self.client = Redis(**config)
        else:
            raise ValueError(
                "Upstash Redis message stores require either a config object or a pre-configured client."
            )

    async def add_message(self, data: HistoryAddMessage) -> None:
        session_id = data.get('sessionId', DEFAULT_CHAT_SESSION_ID)
        session_ttl = data.get('sessionTTL')
        message = data['message']

        await self.client.lpush(session_id, message)

        if session_ttl:
            await self.client.expire(session_id, session_ttl)

    async def delete_messages(self, session_id: str) -> None:
        await self.client.delete(session_id)

    async def get_messages(self, session_id: str = DEFAULT_CHAT_SESSION_ID, amount: int = DEFAULT_HISTORY_LENGTH, start_index: int = 0) -> List[Dict[str, Any]]:
        end_index = start_index + amount - 1

        stored_messages = await self.client.lrange(session_id, start_index, end_index)
        ordered_messages = stored_messages[::-1]
        messages_with_index = [{**message, 'id': str(start_index + index)} for index, message in enumerate(ordered_messages)]

        return messages_with_index