import os
import unittest
from unittest.mock import patch
from .models import openai, upstash, custom
from .ragchat import RAGChat
from langsmith import Client

# Save original environment variables
original_environment = os.environ.copy()

class TestModel(unittest.TestCase):
    
    def test_should_raise_error_when_api_key_is_not_found(self):
        ran = False 

        def throws():
            nonlocal ran
            try:
                RAGChat({
                    'model': upstash("meta-llama/Meta-Llama-3-8B-Instruct", {'apiKey': ""}),
                })
            except Exception as error:
                ran = True
                raise error

        with self.assertRaises(Exception) as context:
            throws()

        self.assertTrue("Failed to create upstash LLM client: QSTASH_TOKEN not found." in str(context.exception))
        self.assertTrue(ran)

class TestModelInits(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(original_environment)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'mock-openai-api-key', 'QSTASH_TOKEN': 'mock-qstash-token'})
    def test_openai_client_configuration_with_helicone(self):
        client = openai("gpt-3.5-turbo", {
            'analytics': {'name': "helicone", 'token': "mock-helicone-token"}
        })

        self.assertEqual(client.clientConfig.baseURL, "https://oai.helicone.ai/v1")
        self.assertEqual(client.clientConfig.defaultHeaders, {
            "Helicone-Auth": "Bearer mock-helicone-token",
            "Authorization": "Bearer mock-openai-api-key",
        })

    def test_openai_client_configuration_without_analytics(self):
        config = {
            'apiKey': "no-key",
            'presencePenalty': 1,
            'maxTokens': 4000,
            'temperature': 0.5,
            'frequencyPenalty': 0.6,
            'topP': 1,
            'logprobs': True,
            'streamUsage': False,
            'topLogprobs': 11,
            'n': 12,
            'logitBias': {},
            'stop': [],
        }

        client = openai("gpt-3.5-turbo", config)

        self.assertEqual(client.clientConfig.baseURL, "https://api.openai.com/v1")
        self.assertEqual(client.clientConfig.apiKey, config['apiKey'])

        # Test all config properties
        for key, value in config.items():
            if key not in ["apiKey", "streamUsage"]:
                self.assertEqual(getattr(client, key), value)

    @patch.dict(os.environ, {'QSTASH_TOKEN': 'mock-qstash-token'})
    def test_upstash_client_configuration(self):
        client = upstash("mistralai/Mistral-7B-Instruct-v0.2", {
            'analytics': {'name': "helicone", 'token': "mock-helicone-token"}
        })

        self.assertEqual(client.clientConfig.baseURL, "https://qstash.helicone.ai/llm/v1")
        self.assertEqual(client.clientConfig.defaultHeaders, {
            "Helicone-Auth": "Bearer mock-helicone-token",
            "Authorization": "Bearer mock-qstash-token",
        })

    def test_custom_client_configuration(self):
        client = custom("custom-model", {
            'baseUrl': "https://custom-llm-api.com",
            'apiKey': "mock-custom-api-key",
            'analytics': {'name': "helicone", 'token': "mock-helicone-token"},
        })

        self.assertEqual(client.clientConfig.baseURL, "https://gateway.helicone.ai")
        self.assertEqual(client.clientConfig.defaultHeaders, {
            "Helicone-Auth": "Bearer mock-helicone-token",
            "Helicone-Target-Url": "https://custom-llm-api.com",
            "Authorization": "Bearer mock-custom-api-key",
        })

    def test_langsmith_analytics_configuration(self):
        with patch('some_module.globalTracer', None):
            openai("gpt-3.5-turbo", {
                'analytics': {'name': "langsmith", 'token': "mock-langsmith-token"}
            })

            self.assertIsNotNone(some_module.globalTracer)
            self.assertIsInstance(some_module.globalTracer, Client)

            some_module.globalTracer = None

    def test_langsmith_analytics_configuration_should_fail_when_token_is_undefined(self):
        with patch('some_module.globalTracer', None):
            openai("gpt-3.5-turbo", {
                'analytics': {'name': "langsmith", 'token': None}
            })

            self.assertFalse(isinstance(some_module.globalTracer, Client))

if __name__ == '__main__':
    unittest.main()
