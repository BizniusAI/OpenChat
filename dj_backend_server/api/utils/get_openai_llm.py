import os
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI, OpenAI

# Load environment variables from a .env file
load_dotenv()

# Constants for environment variables
OPENAI_API_TYPE_KEY = "OPENAI_API_TYPE"
AZURE_OPENAI_API_KEY_KEY = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_DEPLOYMENT_NAME_KEY = "AZURE_OPENAI_DEPLOYMENT_NAME"
AZURE_OPENAI_COMPLETION_MODEL_KEY = "AZURE_OPENAI_COMPLETION_MODEL"
AZURE_OPENAI_API_VERSION_KEY = "AZURE_OPENAI_API_VERSION"
AZURE_OPENAI_API_BASE_KEY = "AZURE_OPENAI_API_BASE"
OPENAI_API_KEY_KEY = "OPENAI_API_KEY"
OPENAI_COMPLETION_MODEL_KEY = "OPENAI_COMPLETION_MODEL"

# Azure OpenAI Language Model client
def get_azure_openai_llm() -> AzureOpenAI:
    """Returns AzureOpenAI instance configured from environment variables"""
    openai_api_type = os.environ[OPENAI_API_TYPE_KEY]
    openai_api_key = os.environ[AZURE_OPENAI_API_KEY_KEY]
    openai_deployment_name = os.environ[AZURE_OPENAI_DEPLOYMENT_NAME_KEY]
    openai_model_name = os.environ[AZURE_OPENAI_COMPLETION_MODEL_KEY]
    openai_api_version = os.environ[AZURE_OPENAI_API_VERSION_KEY]
    openai_api_base = os.environ[AZURE_OPENAI_API_BASE_KEY]
    
    return AzureOpenAI(
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        deployment_name=openai_deployment_name,
        model_name=openai_model_name,
        openai_api_type=openai_api_type,
        openai_api_version=openai_api_version,
        temperature=0,
        batch_size=8
    )

# OpenAI Language Model client
def get_openai_llm() -> OpenAI:
    """Returns OpenAI instance configured from environment variables"""
    openai_api_key = os.environ[OPENAI_API_KEY_KEY]
    openai_completion_model = os.environ[OPENAI_COMPLETION_MODEL_KEY]
    
    return OpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name=openai_completion_model
    )

# Main function to get Language Model client
def get_llm():
    """Returns LLM client instance based on OPENAI_API_TYPE"""
    clients = {
        'azure': get_azure_openai_llm,
        'openai': get_openai_llm
    }
    
    api_type = os.environ.get(OPENAI_API_TYPE_KEY)
    if api_type not in clients:
        raise ValueError(f"Invalid OPENAI_API_TYPE: {api_type}")
    
    return clients[api_type]()
