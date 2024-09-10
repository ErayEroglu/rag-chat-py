from typing import Callable, Dict, List, Optional, Union, Awaitable, ReturnType
from langchain import ChatOpenAI
from openai import openai
from upstash_ratelimit import Ratelimit
from upstash_redis import Redis
from upstash_vector import Index
from .ragchat import CustomPrompt
from langchain import ChatMistralAI # TODO: python equivalent may be different, check requireds


# TODO: what is this about ???
class Brand:
    def __init__(self, brand: str):
        self.__brand = brand

class Branded:
    def __init__(self, value, brand: Brand):
        self.value = value
        self.brand = brand
        
OnChunkType = Callable[[Dict[str, Union[int, str]]], None]
PrepareChatResult = Dict[str, str, any]  # Replace with the actual type if different
OptionalAsyncPrepareChatResult = Union[PrepareChatResult, Awaitable[PrepareChatResult], None, Awaitable[None]]
OnContextFetchedType = Callable[[PrepareChatResult], OptionalAsyncPrepareChatResult]
RatelimitDetailsType = Callable[[Awaitable[ReturnType[Ratelimit["limit"]]]], None]

# Define the type hint for the onChatHistoryFetched callable
UpstashMessage = Dict[str, Union[int, str]]  # Replace with the actual type if different
OptionalAsyncUpstashMessageList = Union[List[UpstashMessage], Awaitable[List[UpstashMessage]], None, Awaitable[None]]
OnChatHistoryFetchedType = Callable[[List[UpstashMessage]], OptionalAsyncUpstashMessageList]

OptionalAsync = Union[object, Awaitable[object]]

class ChatOptions:
    def __init__(self,
                 historyLength: int = 5,
                 historyTTL: int = 86400,
                 similarityThreshold: float = 0.5,
                 topK: int = 5,
                 ratelimitDetails: Optional[RatelimitDetailsType] = None,
                 onChunk: Optional[OnChunkType] = None,
                 onContextFetched: Optional[OnContextFetchedType] = None,
                 onChatHistoryFetched: Optional[OnChatHistoryFetchedType] = None,
                 disableRAG: bool = False):
        self.historyLength = historyLength
        self.historyTTL = historyTTL
        self.similarityThreshold = similarityThreshold
        self.topK = topK
        self.ratelimitDetails = ratelimitDetails
        self.onChunk = onChunk
        self.onContextFetched = onContextFetched
        self.onChatHistoryFetched = onChatHistoryFetched
        self.disableRAG = disableRAG

class PrepareChatResult:
    def __init__(self, data: str, id: str, metadata: object):
        self.data = data
        self.id = id
        self.metadata = metadata

class RAGChatConfig:
    def __init__(self,
                 vector: Optional[Index] = None,
                 redis: Optional[Redis] = None,
                 model: Optional[Union[ChatOpenAI, ChatMistralAI, 'OpenAIChatLanguageModel']] = None,
                 ratelimit: Optional[Ratelimit] = None,
                 debug: bool = False):
        self.vector = vector
        self.redis = redis
        self.model = model
        self.ratelimit = ratelimit
        self.debug = debug

class AddContextOptions:
    def __init__(self, metadata: Optional[Dict] = None, namespace: str = ""):
        self.metadata = metadata
        self.namespace = namespace

class CommonChatAndRAGOptions:
    def __init__(self,
                 streaming: bool = False,
                 sessionId: str = "upstash-rag-chat-session",
                 namespace: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 ratelimitSessionId: str = "upstash-rag-chat-ratelimit-session",
                 promptFn: Optional[CustomPrompt] = None):
        self.streaming = streaming
        self.sessionId = sessionId
        self.namespace = namespace
        self.metadata = metadata
        self.ratelimitSessionId = ratelimitSessionId
        self.promptFn = promptFn

class HistoryOptions:
    def __init__(self, historyLength: int, sessionId: str):
        self.historyLength = historyLength
        self.sessionId = sessionId

class UpstashMessage:
    def __init__(self,
                 role: str,
                 content: str,
                 metadata: Optional[Dict] = None,
                 usage_metadata: Optional[Dict] = None,
                 id: str = ""):
        self.role = role
        self.content = content
        self.metadata = metadata
        self.usage_metadata = usage_metadata
        self.id = id

class OpenAIChatLanguageModel:
    def __init__(self, model : openai):
        self.model = model