import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from api.enums import EmbeddingProvider
from langchain.embeddings.base import Embeddings

# https://github.com/easonlai/azure_openai_langchain_sample/blob/main/chat_with_pdf.ipynb

# Load environment variables from a .env file
load_dotenv()

# Constants for environment variables
EMBEDDING_PROVIDER_KEY = "EMBEDDING_PROVIDER"
AZURE_EMBEDDING_MODEL_NAME_KEY = "AZURE_OPENAI_EMBEDDING_MODEL_NAME"
AZURE_API_KEY_KEY = "AZURE_OPENAI_API_KEY"
AZURE_API_TYPE_KEY = "AZURE_OPENAI_API_TYPE"
AZURE_API_BASE_KEY = "AZURE_OPENAI_API_BASE"
AZURE_API_VERSION_KEY = "AZURE_OPENAI_API_VERSION"
OPENAI_API_KEY_KEY = "OPENAI_API_KEY"

def get_embedding_provider() -> str:
    """Gets the chosen embedding provider from environment variables."""
    return os.environ.get(EMBEDDING_PROVIDER_KEY)

def get_azure_embedding() -> OpenAIEmbeddings:
    """Gets embeddings using the Azure embedding provider."""
    deployment = os.environ.get(AZURE_EMBEDDING_MODEL_NAME_KEY)
    openai_api_key = os.environ.get(AZURE_API_KEY_KEY)
    client = os.environ.get(AZURE_API_TYPE_KEY)
    openai_api_base = os.environ[AZURE_API_BASE_KEY]
    openai_api_version = os.environ[AZURE_API_VERSION_KEY]

    return OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        deployment=deployment,
        client=client,
        chunk_size=8,
        openai_api_base=openai_api_base,
        openai_api_version=openai_api_version
    )

def get_openai_embedding() -> OpenAIEmbeddings:
    """Gets embeddings using the OpenAI embedding provider."""
    openai_api_key = os.environ.get(OPENAI_API_KEY_KEY)

    return OpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=1)

def choose_embedding_provider() -> Embeddings:
    """Chooses and returns the appropriate embedding provider instance."""
    embedding_provider = get_embedding_provider()

    if embedding_provider == EmbeddingProvider.azure.value:
        return get_azure_embedding()
    
    elif embedding_provider == EmbeddingProvider.OPENAI.value:
        return get_openai_embedding()

    else:
        available_providers = ", ".join([service.value for service in EmbeddingProvider])
        raise ValueError(
            f"Embedding service '{embedding_provider}' is not currently available. "
            f"Available services: {available_providers}"
        )

# Main function to get embeddings
def get_embeddings() -> Embeddings:
    """Gets embeddings using the chosen embedding provider."""
    return choose_embedding_provider()
