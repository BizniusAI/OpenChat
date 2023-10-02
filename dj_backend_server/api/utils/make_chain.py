import os
from dotenv import load_dotenv
from langchain.vectorstores.base import VectorStore
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from api.utils.get_openai_llm import get_llm
from api.utils.get_prompts import get_qa_prompt_by_mode

# Load environment variables from a .env file
load_dotenv()

# Constants for environment variables
MODE_KEY = "MODE"
INITIAL_PROMPT_KEY = "INITIAL_PROMPT"

def get_qa_chain(vector_store: VectorStore, mode: str, initial_prompt: str) -> RetrievalQA:
    llm = get_llm()
    template = get_qa_prompt_by_mode(mode, initial_prompt=initial_prompt)
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

def get_retrieval_qa_with_sources_chain(vector_store: VectorStore, mode: str, initial_prompt: str) -> RetrievalQAWithSourcesChain:
    llm = get_llm()
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return chain

def get_conversation_retrieval_chain(vector_store: VectorStore, mode: str, initial_prompt: str) -> ConversationalRetrievalChain:
    llm = get_llm()
    template = get_qa_prompt_by_mode(mode, initial_prompt=initial_prompt)
    prompt = PromptTemplate.from_template(template)

    condense_prompt_template = """
    You are responsible for transforming a chat history into a summary and output summary and exact follow-up input. 
    Please adhere to the following guidelines: 
    1. **Context**: Summarize the chat history. Be as specific as possible in your summary without sacrificing conciseness. 
    2. **Word Limit**: Ensure the summary and the original follow-up input does not exceed 100 words.

    Chat History:
    [[[ {chat_history} ]]]
    Follow-up input: [[[ {question} ]]]
    Summary and the exact follow-up input:
    """
    condense_prompt = PromptTemplate.from_template(template=condense_prompt_template)

    chain = ConversationalRetrievalChain.from_llm(
        llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(), 
        verbose=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain