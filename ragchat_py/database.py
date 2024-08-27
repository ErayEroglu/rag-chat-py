from typing import Any, Dict, List, Optional, Union
from nanoid import generate as nanoid
from constants import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_K
from file_loader import FileDataLoader
from types import AddContextOptions
from upstash_vector import Index

FilePath = str
URL = str

class DatasWithFileSource:
    def __init__(self, type: str, fileSource: Union[FilePath, bytes], options: Optional[AddContextOptions] = None, config: Optional[Dict[str, Any]] = None, pdfConfig: Optional[Dict[str, Any]] = None, csvConfig: Optional[Dict[str, Any]] = None, htmlConfig: Optional[Dict[str, Any]] = None):
        self.type = type
        self.fileSource = fileSource
        self.options = options
        self.config = config
        self.pdfConfig = pdfConfig
        self.csvConfig = csvConfig
        self.htmlConfig = htmlConfig

class AddContextPayload:
    def __init__(self, type: str, data: Union[str, List[float], DatasWithFileSource], options: Optional[AddContextOptions] = None, id: Optional[Union[str, int]] = None):
        self.type = type
        self.data = data
        self.options = options
        self.id = id

class VectorPayload:
    def __init__(self, question: str, similarityThreshold: float = DEFAULT_SIMILARITY_THRESHOLD, topK: int = DEFAULT_TOP_K, namespace: Optional[str] = None):
        self.question = question
        self.similarityThreshold = similarityThreshold
        self.topK = topK
        self.namespace = namespace

class ResetOptions:
    def __init__(self, namespace: str):
        self.namespace = namespace

class SaveOperationResult:
    def __init__(self, success: bool, ids: Optional[List[str]] = None, error: Optional[str] = None):
        self.success = success
        self.ids = ids
        self.error = error

class Database:
    def __init__(self, index: Index):
        self.index = index

    async def reset(self, options: Optional[ResetOptions] = None):
        await self.index.reset({"namespace": options.namespace if options else None})

    async def delete(self, ids: List[str], namespace: Optional[str] = None):
        await self.index.delete(ids, {"namespace": namespace})
    
    """
    A method that allows you to query the vector database with plain text.
    It takes care of the text-to-embedding conversion by itself.
    Additionally, it lets consumers pass various options to tweak the output.
    """
    async def retrieve(self, payload: VectorPayload) -> List[Dict[str, Any]]:
        result = await self.index.query(
            {
                "data": payload.question,
                "topK": payload.topK,
                "includeData": True,
                "includeMetadata": True,
            },
            {"namespace": payload.namespace}
        )
        all_values_undefined = all(embedding["data"] is None for embedding in result)

        if all_values_undefined:
            print("There is no answer for this question in the provided context.")
            return [{"data": "There is no answer for this question in the provided context.", "id": "error", "metadata": {}}]

        facts = [
            {
                "data": f"- {embedding['data'] or ''}",
                "id": str(embedding["id"]),
                "metadata": embedding["metadata"],
            }
            for embedding in result if embedding["score"] >= payload.similarityThreshold
        ]

        return facts

    async def save(self, input: AddContextPayload) -> SaveOperationResult:
        namespace = input.options.namespace if input.options else None
        try:
            if input.type == "text":
                vector_id = await self.index.upsert(
                    {
                        "data": input.data,
                        "id": input.id or nanoid(),
                        "metadata": input.options.metadata if input.options else None,
                    },
                    {"namespace": namespace}
                )
                return SaveOperationResult(success=True, ids=[str(vector_id)])
            elif input.type == "embedding":
                vector_id = await self.index.upsert(
                    {
                        "vector": input.data,
                        "id": input.id or nanoid(),
                        "metadata": input.options.metadata if input.options else None,
                    },
                    {"namespace": namespace}
                )
                return SaveOperationResult(success=True, ids=[str(vector_id)])
            else:
                file_args = input.pdfConfig if hasattr(input, 'pdfConfig') else input.csvConfig if hasattr(input, 'csvConfig') else {}
                transform_or_split = await FileDataLoader(input).load_file(file_args)
                transform_args = input.config if hasattr(input, 'config') else {}
                transform_documents = await transform_or_split(transform_args)
                await self.index.upsert(transform_documents, {"namespace": namespace})
                return SaveOperationResult(success=True, ids=[str(doc["id"]) for doc in transform_documents])
        except Exception as error:
            print(error)
            return SaveOperationResult(success=False, error=str(error))