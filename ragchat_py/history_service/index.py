from typing import Optional, Dict, Any, TypedDict
from redis import Redis
from .in_memory_history import InMemoryHistory
from .redis_custom_history import UpstashRedisHistory

class HistoryConfig:
    def __init__(self, redis: Optional['Redis'] = None):
        self.redis = redis

class GetHistoryOptions(TypedDict):
    sessionId: str
    length: Optional[int]
    sessionTTL: Optional[int]

class HistoryService:
    def __init__(self, fields: Optional[HistoryConfig] = None):
        if fields and 'redis' in fields and fields['redis']:
            self.service = UpstashRedisHistory(client=fields['redis'])
        else:
            self.service = InMemoryHistory()