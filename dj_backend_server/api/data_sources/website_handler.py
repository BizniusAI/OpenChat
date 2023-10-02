import os
import traceback
from django.http import JsonResponse
from langchain.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from api.utils import init_vector_store
from api.utils.get_embeddings import get_embeddings
from api.interfaces import StoreOptions
from web.models.website_data_sources import WebsiteDataSource
from web.enums.website_data_source_status_enum import WebsiteDataSourceStatusType

def website_handler(shared_folder, namespace):
    try:
        # Get the website data source by ID
        website_data_source = WebsiteDataSource.objects.get(id=shared_folder)

        # Define the directory path
        directory_path = os.path.join("website_data_sources", shared_folder)

        # Load text documents from the specified directory
        directory_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
        raw_docs = directory_loader.load()
        print('Loaded text documents')

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        docs = text_splitter.split_documents(raw_docs)
        print("docs -->", docs)

        # Get word embeddings
        embeddings = get_embeddings()

        # Initialize the vector store
        init_vector_store(docs, embeddings, StoreOptions(namespace=namespace))

        # Update the website data source status to COMPLETED
        website_data_source.crawling_status = WebsiteDataSourceStatusType.COMPLETED.value
        website_data_source.save()

        # Uncomment the following line to delete the folder (if needed)
        # delete_folder(folder_path=directory_path)
        print('All done. Folder deleted...')
    except Exception as e:
        # Update the website data source status to FAILED
        website_data_source.crawling_status = WebsiteDataSourceStatusType.FAILED.value
        website_data_source.save()

        # Print and log the exception
        print(e)
        traceback.print_exc()
