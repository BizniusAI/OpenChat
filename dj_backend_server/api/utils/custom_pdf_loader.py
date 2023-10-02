import io
from langchain.docstore.base import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import PyPDFLoader

class BufferLoader(BaseLoader):
    """
    A document loader that loads documents from a file path or Blob.
    """

    def __init__(self, filePathOrBlob):
        """
        Initialize the BufferLoader with the provided file path or Blob.

        Args:
            filePathOrBlob: A file path (str) or a Blob object.
        """
        super().__init__()
        self.filePathOrBlob = filePathOrBlob

    async def load(self):
        """
        Load and parse the document.

        Returns:
            A list of Document objects.
        """
        buffer, metadata = None, {}
        if isinstance(self.filePathOrBlob, str):
            with open(self.filePathOrBlob, 'rb') as file:
                buffer = file.read()
            metadata = {'source': self.filePathOrBlob}
        else:
            buffer = await self.filePathOrBlob.read()
            metadata = {'source': 'blob', 'blobType': self.filePathOrBlob.type}
        return await self.parse(buffer, metadata)

    async def parse(self, raw, metadata):
        """
        Parse the raw document data and metadata.

        Args:
            raw: Raw document data.
            metadata: Metadata associated with the document.

        Returns:
            A list of Document objects.
        """
        raise NotImplementedError("The 'parse' method must be implemented in subclasses.")

class CustomPDFLoader(BufferLoader):
    """
    A custom PDF document loader that extracts text content from PDFs.
    """

    async def parse(self, raw, metadata):
        """
        Parse the raw PDF data and extract text content.

        Args:
            raw: Raw PDF data.
            metadata: Metadata associated with the PDF.

        Returns:
            A list of Document objects containing extracted text content.
        """
        parsed = self.parse_pdf(raw)
        return [
            Document(pageContent=parsed['text'], metadata={
                **metadata, 'pdf_numpages': parsed['numpages'],
            })
        ]

    def parse_pdf(self, raw):
        """
        Parse a PDF and extract text content.

        Args:
            raw: Raw PDF data.

        Returns:
            A dictionary with 'text' (extracted text) and 'numpages' (number of pages).
        """
        pdf_loader = PyPDFLoader(io.BytesIO(raw))
        text = ""
        num_pages = pdf_loader.num_pages()
        for page_num in range(num_pages):
            text += pdf_loader.extract_text(page_num)
        return {'text': text, 'numpages': num_pages}

# Usage example:
# Replace 'filePathOrBlobValue' with the actual file path or Blob object.
# filePathOrBlobValue = "/path/to/pdf_file.pdf"
# loader = CustomPDFLoader(filePathOrBlobValue)
# docs = await loader.load()