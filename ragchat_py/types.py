from typing import Callable, Optional, Union, List, Dict, Any, Awaitable
from pydantic import BaseModel

OptionalAsync = Union[Awaitable[Any], Any]

# Constants for default values
DEFAULT_HISTORY_LENGTH = 5
DEFAULT_HISTORY_TTL = 86_400  # 1 day in seconds
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_TOP_K = 5

# PrepareChatResult model
class PrepareChatResult(BaseModel):
    data: str
    id: str
    metadata: Any

# UpstashDict type for metadata
UpstashDict = Dict[str, Any]

# UpstashMessage type
class UpstashMessage(BaseModel):
    role: str  # "assistant" or "user"
    content: str
    metadata: Optional[UpstashDict] = None
    usage_metadata: Optional[Dict[str, int]] = None
    id: str

# ChatOptions class
class ChatOptions(BaseModel):
    # Length of the conversation history to include in your LLM query. Increasing this may lead to hallucinations. Retrieves the last N messages.
    # @default 5
    history_length: Optional[int] = DEFAULT_HISTORY_LENGTH

    # Configuration to retain chat history. After the specified time, the history will be automatically cleared.
    # @default 86_400 // 1 day in seconds
    history_ttl: Optional[int] = DEFAULT_HISTORY_TTL

    # Configuration to adjust the accuracy of results.
    # @default 0.5
    similarity_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD

    # Amount of data points to include in your LLM query.
    # @default 5
    top_k: Optional[int] = DEFAULT_TOP_K

    # Details of applied rate limit.
    ratelimit_details: Optional[Callable[[Dict[str, Any]], None]] = None

    # Hook to modify or get data and details of each chunk. Can be used to alter streamed content.
    on_chunk: Optional[Callable[[Dict[str, Any]], None]] = None

    # Hook to access the retrieved context and modify as you wish.
    on_context_fetched: Optional[Callable[[List[PrepareChatResult]], OptionalAsync[Union[List[PrepareChatResult], None]]]] = None

    # Hook to access the retrieved history and modify as you wish.
    on_chat_history_fetched: Optional[Callable[[List[UpstashMessage]], OptionalAsync[Union[List[UpstashMessage], None]]]] = None

    # Allows disabling RAG and use chat as LLM in combination with prompt. This will give you ability to build your own pipelines.
    disable_rag: Optional[bool] = False

# CommonChatAndRAGOptions class
class CommonChatAndRAGOptions(BaseModel):
    # Set to `true` if working with web apps and you want to be interactive without stalling users.
    streaming: Optional[bool] = False

    # Chat session ID of the user interacting with the application.
    # @default "upstash-rag-chat-session"
    session_id: Optional[str] = "upstash-rag-chat-session"

    # Namespace of the index you wanted to query.
    namespace: Optional[str] = None

    # Metadata for your chat message. This could be used to store anything in the chat history. By default RAG Chat SDK uses this to persist used model name in the history
    metadata: Optional[UpstashDict] = None

    # Rate limit session ID of the user interacting with the application.
    # @default "upstash-rag-chat-ratelimit-session"
    ratelimit_session_id: Optional[str] = "upstash-rag-chat-ratelimit-session"

    # If no Index name or instance is provided, falls back to the default.
    # @default
    # PromptTemplate.fromTemplate(`You are a friendly AI assistant augmented with an Upstash Vector Store.
    # To help you answer the questions, a context will be provided. This context is generated by querying the vector store with the user question.
    # Answer the question at the end using only the information available in the context and chat history.
    # If the answer is not available in the chat history or context, do not answer the question and politely let the user know that you can only answer if the answer is available in context or the chat history.
    #
    # -------------
    # Chat history:
    # {chat_history}
    # -------------
    # Context:
    # {context}
    # -------------
    #
    # Question: {question}
    # Helpful answer:`)
    prompt_fn: Optional[Callable[[str], str]] = None

# RAGChatConfig class
class RAGChatConfig(CommonChatAndRAGOptions):
    # Assuming Index is a class
    vector: Optional[Any] = None

    # Assuming Redis is a class
    redis: Optional[Any] = None

    # ChatOpenAI, ChatMistralAI, OpenAIChatLanguageModel
    model: Optional[Union[Any, Any, Any]] = None

    # Assuming Ratelimit is a class
    ratelimit: Optional[Any] = None

    # Logs every step of the chat, including sending prompts, listing history entries,
    # retrieving context from the vector database, and capturing the full response
    # from the LLM, including latency.
    debug: Optional[bool] = False

# AddContextOptions class
class AddContextOptions(BaseModel):
    # Namespace of the index you wanted to insert. Default is empty string.
    # @default ""
    metadata: Optional[UpstashDict] = None
    namespace: Optional[str] = ""

# HistoryOptions class
class HistoryOptions(BaseModel):
    history_length: Optional[int] = None
    session_id: Optional[str] = None