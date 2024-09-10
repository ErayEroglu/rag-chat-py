import pytest
import asyncio
from database import Database
from upstash_vector import Index
from .test_utils import await_until_indexed

@pytest.fixture(scope="module")
async def vector():
    vector = Index()
    yield vector
    await vector.reset()

@pytest.mark.asyncio
async def test_save_and_retrieve_info_using_data_field(vector):
    database = Database(vector)
    await database.save({
        'type': 'text',
        'data': 'Paris, the capital of France, is renowned for its iconic landmark, the Eiffel Tower, which was completed in 1889 and stands at 330 meters tall.',
    })
    await database.save({
        'type': 'text',
        'data': 'The city is home to numerous world-class museums, including the Louvre Museum, housing famous works such as the Mona Lisa and Venus de Milo.',
    })
    await database.save({
        'type': 'text',
        'data': 'Paris is often called the City of Light due to its significant role during the Age of Enlightenment and its early adoption of street lighting.',
    })
    await await_until_indexed(vector)

    result = await database.retrieve({
        'question': 'What year was the construction of the Eiffel Tower completed, and what is its height?',
        'topK': 1,
        'similarityThreshold': 0.5,
        'namespace': '',
    })
    assert '330' in ' '.join([item['data'] for item in result])