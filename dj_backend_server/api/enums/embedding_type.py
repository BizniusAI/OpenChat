from enum import Enum

class EmbeddingProvider(Enum):
    """
    Enumeration of available embedding providers.
    """
    OPENAI = "openai"
    """
    OpenAI's GPT-3 model for generating embeddings.
    """
    BARD = "bard"
    """
    Bard's embedding provider.
    """
    AZURE = "azure"
    """
    Microsoft Azure's embedding provider.
    """
