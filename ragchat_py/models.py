from typing import List, Optional, Dict, Union
from langchain.chat_models.openai import ChatOpenAI
from langsmith import Client as LangsmithClient
import requests
from .constants import OLLAMA_MODELS

globalTracer: Optional[LangsmithClient] = None

class OpenAIChatModel:
    GPT_MODELS = [
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview",
        "gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-vision-preview",
        "gpt-4", "gpt-4o", "gpt-4-0314", "gpt-4-0613", "gpt-4-32k",
        "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-3.5-turbo", 
        "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k-0613"
    ]

class UpstashChatModel:
    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2", 
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]

class LLMClientConfig:
    def __init__(self,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 n: Optional[int] = None,
                 logit_bias: Optional[Dict[str, float]] = None,
                 model: str = '',
                 model_kwargs: Optional[dict] = None,
                 stop: Optional[list] = None,
                 stop_sequences: Optional[list] = None,
                 user: Optional[str] = None,
                 timeout: Optional[int] = None,
                 stream_usage: Optional[bool] = None,
                 max_tokens: Optional[int] = None,
                 logprobs: Optional[bool] = None,
                 top_logprobs: Optional[int] = None,
                 openai_api_key: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: str = ''):
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n
        self.logit_bias = logit_bias
        self.model = model
        self.model_kwargs = model_kwargs
        self.stop = stop
        self.stop_sequences = stop_sequences
        self.user = user
        self.timeout = timeout
        self.stream_usage = stream_usage
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.openai_api_key = openai_api_key
        self.api_key = api_key
        self.base_url = base_url


class HeliconeConfig:
    def __init__(self, token: str):
        self.name = "helicone"
        self.token = token

class LangsmithConfig:
    def __init__(self, token: str, apiUrl: Optional[str] = None):
        self.name = "langsmith"
        self.token = token
        self.apiUrl = apiUrl
        
class AnalyticsConfig:
    def __init__(self, config: Union[HeliconeConfig, LangsmithConfig]):
        self.config = config
        
class AnalyticsSetup:
    def __init__(self, base_url: Optional[str] = None,
                 default_headers: Optional[Dict[str, Optional[str]]] = None,
                 client: Optional['LangsmithClient'] = None):
        self.base_url = base_url
        self.default_headers = default_headers
        self.client = client
        
class ModelOptions:
    def __init__(self, 
                 baseUrl: str,
                 temperature: Optional[float] = None,
                 topP: Optional[float] = None,
                 frequencyPenalty: Optional[float] = None,
                 presencePenalty: Optional[float] = None,
                 n: Optional[int] = None,
                 logitBias: Optional[Dict[str, float]] = None,
                 modelKwargs: Optional[Dict] = None,
                 stop: Optional[list] = None,
                 stopSequences: Optional[list] = None,
                 user: Optional[str] = None,
                 timeout: Optional[int] = None,
                 streamUsage: Optional[bool] = None,
                 maxTokens: Optional[int] = None,
                 logprobs: Optional[bool] = None,
                 topLogprobs: Optional[int] = None,
                 openAIApiKey: Optional[str] = None,
                 apiKey: Optional[str] = None,
                 analytics: Optional[AnalyticsConfig] = None):
        self.temperature = temperature
        self.topP = topP
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.n = n
        self.logitBias = logitBias
        self.modelKwargs = modelKwargs
        self.stop = stop
        self.stopSequences = stopSequences
        self.user = user
        self.timeout = timeout
        self.streamUsage = streamUsage
        self.maxTokens = maxTokens
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.openAIApiKey = openAIApiKey
        self.apiKey = apiKey
        self.baseUrl = baseUrl
        self.analytics = analytics

Providers = [
    "openai",
    "upstash",
    "custom",
    "ollama",
    "groq",
    "togetherai",
    "openrouter",
    "mistral"
]

def setup_analytics(analytics: Optional[AnalyticsConfig],
                    provider_api_key: str,
                    provider_base_url: Optional[str] = None,
                    provider: Optional[str] = None) -> AnalyticsSetup:
    if not analytics:
        return {}

    if analytics.name == "helicone":
        if provider == "openai":
            return AnalyticsSetup(
                base_url="https://oai.helicone.ai/v1",
                default_headers={
                    "Helicone-Auth": f"Bearer {analytics.token}",
                    "Authorization": f"Bearer {provider_api_key}",
                }
            )
        elif provider == "upstash":
            return AnalyticsSetup(
                base_url="https://qstash.helicone.ai/llm/v1",
                default_headers={
                    "Helicone-Auth": f"Bearer {analytics.token}",
                    "Authorization": f"Bearer {provider_api_key}",
                }
            )
        else:
            return AnalyticsSetup(
                base_url="https://gateway.helicone.ai",
                default_headers={
                    "Helicone-Auth": f"Bearer {analytics.token}",
                    "Helicone-Target-Url": provider_base_url,
                    "Authorization": f"Bearer {provider_api_key}",
                }
            )
    elif analytics.name == "langsmith":
        if analytics.token!= None and analytics.token != "":
            client = LangsmithClient(api_key=analytics.token, api_url=analytics.api_url or "https://api.smith.langchain.com")
            global globalTracer
            globalTracer = client
            return AnalyticsSetup(client=client)
        return AnalyticsSetup(client=None)

    else:
        raise ValueError(f"Unsupported analytics provider: {analytics.name}")


def create_llm_client(model: str, options: ModelOptions, provider: Optional[str] = None):
    api_key = options.api_key or ""
    provider_base_url = options.base_url
    if not api_key:
        raise ValueError("API key is required. Provide it in options or set environment variable.")

    analytics = options.analytics
    analytics_setup = setup_analytics(analytics, api_key, provider_base_url, provider)
    
    rest_options = {k : v for k, v in options.__dict__.items() if k != "analytics"}
    return ChatOpenAI(
        model_name=model,
        stream_usage=provider != "upstash",
        temperature=options.temperature or 0,
        api_key=api_key,
        rest_options=rest_options,
        configuration={
            "base_url": analytics_setup.base_url or provider_base_url,
            **({"default_headers": analytics_setup.default_headers} if analytics_setup.default_headers else {}),
        }
    )

def upstash(model: str, options: Optional[ModelOptions] = None):
    api_key = options.api_key or ""
    if not api_key:
        raise ValueError("Failed to create upstash LLM client: QSTASH_TOKEN not found." +
                         "Pass apiKey parameter or set QSTASH_TOKEN env variable.")
    return create_llm_client(model, options, "upstash")

def custom(model: str, options: ModelOptions):
    if not options.base_url:
        raise ValueError("base_url cannot be empty or undefined.")
    return create_llm_client(model, options, "custom")

def openai(model: str, options: Optional[ModelOptions] = None):
    modified_options = {**options, "base_url": "https://api.openai.com/v1"}
    return create_llm_client(model, modified_options, "openai")

def groq(model: str, options: Optional[LLMClientConfig] = None):
    modified_options = {**options, "base_url": "https://api.groq.com/openai/v1"}
    return create_llm_client(model, modified_options, provider="groq")

def togetherai(model: str, options: Optional[LLMClientConfig] = None):
    modified_options = {**options, "base_url": "https://api.together.xyz/v1"}
    return create_llm_client(model, modified_options, provider="togetherai")

def openrouter(model: str, options: Optional[LLMClientConfig] = None):
    modified_options = {**options, "base_url": "https://openrouter.ai/api/v1"} 
    return create_llm_client(model, modified_options, provider="openrouter")

class OllamaModelResult:
    def __init__(self, models: List[Dict[str, str]]):
        self.models = models

def ollama(model, options : any):
    DEFAULT_OLLAMA_PORT = 11_434
    if options is None:
        options = {}
    
    port = options.get('port', DEFAULT_OLLAMA_PORT)
    base_url = f"http://localhost:{port}"

    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        
        model_exists = any(m['name'].find(model) != -1 for m in data['models'])
        if not model_exists:
            print(f"Model not found. Please pull the model before running.\nRun: ollama pull {model}")
    except requests.RequestException as error:
        print("Error checking model availability:", error)
    return create_llm_client(model, {**options, 'base_url': f"{base_url}/v1"}, "ollama")