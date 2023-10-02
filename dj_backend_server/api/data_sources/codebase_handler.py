from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from api.utils import get_embeddings, init_vector_store
from langchain.document_loaders import GitLoader
from api.interfaces import StoreOptions

# https://python.langchain.com/docs/integrations/document_loaders/git
@csrf_exempt
def codebase_handler(repo_path: str, namespace: str):
    try:
        folder_path = f"website_data_sources/{namespace}"

        # Load documents from Git repository
        loader = GitLoader(repo_path=folder_path, clone_url=repo_path, branch="master")
        raw_docs = loader.load()
        print('Loaded documents')

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1000, chunk_overlap=200, length_function=len)
        docs = text_splitter.split_documents(raw_docs)

        # Get word embeddings
        embeddings = get_embeddings()

        # Initialize the vector store
        init_vector_store(docs, embeddings, options=StoreOptions(namespace))
        print('Indexed documents. All done!')
    except Exception as e:
        print(e)
