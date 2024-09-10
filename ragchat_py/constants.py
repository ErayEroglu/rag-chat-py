
# import CustomPrompt from .ragchat

DEFAULT_CHAT_SESSION_ID = "upstash-rag-chat-session"
DEFAULT_CHAT_RATELIMIT_SESSION_ID = "upstash-rag-chat-ratelimit-session"

RATELIMIT_ERROR_MESSAGE = "ERR:USER_RATELIMITED"

DEFAULT_VECTOR_DB_NAME = "upstash-rag-chat-vector"
DEFAULT_REDIS_DB_NAME = "upstash-rag-chat-redis"

# Retrieval related default options
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_TOP_K = 5

# History related default options
DEFAULT_HISTORY_TTL = 86_400
DEFAULT_HISTORY_LENGTH = 5

# We need that constant to split creator LLM such as `ChatOpenAI_gpt-3.5-turbo`. Format is `provider_modelName`.
MODEL_NAME_WITH_PROVIDER_SPLITTER = "_"

# We need to make sure namespace is not undefined, but "". This will force vector-db to query default namespace.
DEFAULT_NAMESPACE = ""

def default_prompt(context: str, question: str, chat_history: str) -> str:
    return f"""You are a friendly AI assistant augmented with an Upstash Vector Store.
To help you answer the questions, a context and/or chat history will be provided.
Answer the question at the end using only the information available in the context or chat history, either one is ok.

-------------
Chat history:
{chat_history}
-------------
Context:
{context}
-------------

Question: {question}
Helpful answer:"""

def DEFAULT_PROMPT_WITHOUT_RAG(question: str, chat_history: str) -> str:
    return f"""You are a friendly AI assistant.
To help you answer the questions, a chat history will be provided.
Answer the question at the end.
-------------
Chat history:
{chat_history}
-------------
Question: {question}
Helpful answer:"""

OLLAMA_MODELS = [
    "llama3.1", "gemma2", "mistral-nemo", "mistral-large", "qwen2", "deepseek-coder-v2", "phi3", "mistral", "mixtral",
    "codegemma", "command-r", "command-r-plus", "llava", "llama3", "gemma", "qwen", "llama2", "codellama", "dolphin-mixtral",
    "nomic-embed-text", "llama2-uncensored", "phi", "deepseek-coder", "zephyr", "mxbai-embed-large", "dolphin-mistral",
    "orca-mini", "dolphin-llama3", "starcoder2", "yi", "mistral-openorca", "llama2-chinese", "llava-llama3", "starcoder",
    "vicuna", "tinyllama", "codestral", "wizard-vicuna-uncensored", "nous-hermes2", "wizardlm2", "openchat", "aya",
    "tinydolphin", "stable-code", "wizardcoder", "openhermes", "all-minilm", "granite-code", "codeqwen", "stablelm2",
    "wizard-math", "neural-chat", "phind-codellama", "llama3-gradient", "dolphincoder", "nous-hermes", "sqlcoder", "xwinlm",
    "deepseek-llm", "yarn-llama2", "llama3-chatqa", "starling-lm", "wizardlm", "falcon", "orca2", "snowflake-arctic-embed",
    "solar", "samantha-mistral", "moondream", "stable-beluga", "dolphin-phi", "bakllava", "deepseek-v2", "wizardlm-uncensored",
    "yarn-mistral", "medllama2", "llama-pro", "glm4", "nous-hermes2-mixtral", "meditron", "codegeex4", "nexusraven", "llava-phi3",
    "codeup", "everythinglm", "magicoder", "stablelm-zephyr", "codebooga", "mistrallite", "wizard-vicuna", "duckdb-nsql",
    "megadolphin", "falcon2", "notux", "goliath", "open-orca-platypus2", "notus", "internlm2", "llama3-groq-tool-use", "dbrx",
    "alfred", "mathstral", "firefunction-v2", "nuextract", "bge-m3", "bge-large", "paraphrase-multilingual"
]