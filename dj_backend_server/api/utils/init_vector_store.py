import os
import threading
from dotenv import load_dotenv
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores.qdrant import Qdrant
from api.enums import StoreType
from api.interfaces import StoreOptions
from api.configs import PINECONE_TEXT_KEY, VECTOR_STORE_INDEX_NAME

# Load environment variables from a .env file
load_dotenv()

# Constants for environment variables
STORE_TYPE_KEY = "STORE"
QDRANT_URL_KEY = "QDRANT_URL"
PINECONE_API_KEY_KEY = "PINECONE_API_KEY"
PINECONE_ENV_KEY = "PINECONE_ENV"

init_lock = threading.Lock()
initialized = False

def initialize_pinecone():
    global initialized
    # Only initialize Pinecone if the store type is Pinecone and the initialization lock is not acquired
    with init_lock:
        if not initialized:
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv(PINECONE_API_KEY_KEY),  # find at app.pinecone.io
                environment=os.getenv(PINECONE_ENV_KEY),  # next to api key in console
            )
            initialized = True

def init_vector_store(docs: list[Document], embeddings: OpenAIEmbeddings, options: StoreOptions) -> None:
    store_type = StoreType[os.environ[STORE_TYPE_KEY]]

    if store_type == StoreType.PINECONE:
        initialize_pinecone()

        # Use the Pinecone vector store
        Pinecone.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=VECTOR_STORE_INDEX_NAME,
            namespace=options.namespace,
            text_key=PINECONE_TEXT_KEY
        )

    elif store_type == StoreType.QDRANT:
        Qdrant.from_documents(
            docs=docs,
            embeddings=embeddings,
            collection_name=options.namespace,
            url=os.environ[QDRANT_URL_KEY]
        )

    else:
        valid_stores = ", ".join(StoreType._member_names())
        raise ValueError(f"Invalid STORE environment variable value: {os.environ[STORE_TYPE_KEY]}. Valid values are: {valid_stores}")
