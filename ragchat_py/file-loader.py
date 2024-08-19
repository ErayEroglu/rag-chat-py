import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from langchain.document_loaders.fs.csv import CSVLoader
from langchain.document_loaders.fs.pdf import PDFLoader
from langchain.document_loaders.web.cheerio import CheerioWebBaseLoader
from langchain.document_transformers.html_to_text import HtmlToTextTransformer
from langchain.core.documents import Document
from langchain.document_loaders.fs.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nanoid import generate as nanoid
from database import DatasWithFileSource, FilePath, URL

class FileDataLoader:
    def __init__(self, config: DatasWithFileSource):
        self.config = config

    async def load_file(self, args: Any):
        loader = self.create_loader(args)
        documents = await loader.load()
        return lambda args: self.transform_document(documents, args)

    def create_loader(self, args: Any):
        if self.config['type'] == "pdf":
            return PDFLoader(self.config['fileSource'], args)
        elif self.config['type'] == "csv":
            return CSVLoader(self.config['fileSource'], args)
        elif self.config['type'] == "text-file":
            return TextLoader(self.config['fileSource'])
        elif self.config['type'] == "html":
            return CheerioWebBaseLoader(self.config['source']) if self.is_url(self.config['source']) else TextLoader(self.config['source'])
        else:
            raise ValueError(f"Unsupported data type: {self.config['type']}")

    def is_url(self, source: Union[FilePath, Any]) -> bool: 
        return isinstance(source, str) and source.startswith("http")

    async def transform_document(self, documents: List[Document], args: Any):
        if self.config['type'] == "pdf":
            splitter = RecursiveCharacterTextSplitter(args)
            splitted_documents = await splitter.split_documents(documents)
            return self.map_documents_into_insert_payload(splitted_documents, lambda metadata, index: {
                "source": metadata.get("source"),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "paragraphNumber": index + 1,
                "pageNumber": metadata.get("loc", {}).get("pageNumber"),
                "author": metadata.get("pdf", {}).get("info", {}).get("Author"),
                "title": metadata.get("pdf", {}).get("info", {}).get("Title"),
                "totalPages": metadata.get("pdf", {}).get("totalPages"),
                "language": metadata.get("pdf", {}).get("metadata", {}).get("_metadata", {}).get("dc:language")
            })
        elif self.config['type'] == "csv":
            return self.map_documents_into_insert_payload(documents)
        elif self.config['type'] == "text-file":
            splitter = RecursiveCharacterTextSplitter(args)
            splitted_documents = await splitter.split_documents(documents)
            return self.map_documents_into_insert_payload(splitted_documents)
        elif self.config['type'] == "html":
            splitter = RecursiveCharacterTextSplitter.from_language("html", args)
            transformer = HtmlToTextTransformer()
            sequence = splitter.pipe(transformer)
            new_documents = await sequence.invoke(documents)
            return self.map_documents_into_insert_payload(new_documents)
        else:
            raise ValueError(f"Unsupported data type: {self.config['type']}")

    def map_documents_into_insert_payload(
        self, 
        splitted_documents: List[Document], 
        metadata_mapper: Optional[Callable[[Any, int], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:        
        return [
                {
                    "data": document.page_content,
                    "id": nanoid(),
                    **({"metadata": metadata_mapper(document.metadata, index)} if metadata_mapper else {})
                }
                for index, document in enumerate(splitted_documents)
        ]