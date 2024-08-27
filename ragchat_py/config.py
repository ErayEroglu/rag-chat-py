import os
from typing import Optional, Dict, Union
from upstash_redis import Redis
from upstash_vector import Index
from upstash_ratelimit import Ratelimit
from models import upstash, openai
from constants import DEFAULT_PROMPT
from types import RAGChatConfig, UpstashDict, OpenAIChatLanguageModel, CustomPrompt


class Config:
    def __init__(self, config: Optional[RAGChatConfig] = None):
        # Initialize redis instances
        self.redis = config.redis if config and config.redis else self.initialize_redis()

        # Rate limiting settings
        self.ratelimit = config.ratelimit if config else None
        self.ratelimit_session_id = config.ratelimit_session_id if config else None

        # Chat configuration settings
        self.streaming = config.streaming if config else None
        self.namespace = config.namespace if config else None
        self.metadata = config.metadata if config else None
        self.session_id = config.session_id if config else None

        # Initialize model
        self.model = config.model if config and config.model else self.initialize_model()

        # Custom prompt setup
        self.prompt = config.prompt_fn if config and config.prompt_fn else DEFAULT_PROMPT

        # Initialize vector index
        self.vector = config.vector if config and config.vector else Index.from_env()

        # Debug flag
        self.debug = config.debug if config else None

    @staticmethod
    def initialize_redis() -> Optional[Redis]:
        try:
            return Redis.from_env()
        except Exception as e:
            # Handle the exception if environment variables for Redis are missing
            return None

    @staticmethod
    def initialize_model() -> Union[OpenAIChatLanguageModel, None]:
        qstash_token = os.getenv("QSTASH_TOKEN")
        openai_token = os.getenv("OPENAI_API_KEY")

        if qstash_token:
            return upstash("meta-llama/Meta-Llama-3-8B-Instruct", api_key=qstash_token)

        if openai_token:
            return openai("gpt-4o", api_key=openai_token)

        raise ValueError(
            "[RagChat Error]: Unable to connect to model. "
            "Pass one of OPENAI_API_KEY or QSTASH_TOKEN environment variables."
        )
