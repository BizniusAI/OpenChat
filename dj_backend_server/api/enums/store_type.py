from enum import Enum

class StoreType(Enum):
    """
    Enumeration of available store types.
    """
    PINECONE = 'PINECONE'
    """
    Pinecone vector store.
    """
    QDRANT = 'QDRANT'
    """
    Qdrant vector store.
    """
