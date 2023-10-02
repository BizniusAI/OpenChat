import json
import os
import traceback
from django.views.decorators.csrf import csrf_exempt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from api.utils import get_embeddings, init_vector_store
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders import PyPDFium2Loader
from web.utils.delete_foler import delete_folder
from api.interfaces import StoreOptions

@csrf_exempt
def pdf_handler(shared_folder: str, namespace: str):
    try:
        # Define the directory path
        directory_path = os.path.join("website_data_sources", shared_folder)

        # Load PDF documents from the specified directory
        directory_loader = DirectoryLoader(path=directory_path, glob="**/*.pdf", loader_cls=PyPDFium2Loader, use_multithreading=True)
        raw_docs = directory_loader.load_and_split()
        print('Loaded PDF documents')

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        docs = text_splitter.split_documents(raw_docs)

        # Get word embeddings
        embeddings = get_embeddings()

        # Initialize the vector store
        init_vector_store(docs, embeddings, StoreOptions(namespace))

        # Delete the folder after processing
        delete_folder(folder_path=directory_path)
        print('All done. Folder deleted.')

    except Exception as e:
        print(e)
        traceback.print_exc()
