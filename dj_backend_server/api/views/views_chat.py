from django.http import JsonResponse
from django.views.decorators.http import require_POST
from langchain import QAWithSourcesChain

from api.utils import get_vector_store
from api.utils.make_chain import getConversationRetrievalChain, getRetrievalQAWithSourcesChain
import json
from django.views.decorators.csrf import csrf_exempt
from api.interfaces import StoreOptions
from web.models.chat_histories import ChatHistory
from django.shortcuts import get_object_or_404
from web.models.chatbot import Chatbot
from uuid import uuid4
import logging
import traceback
from web.services.chat_history_service import get_chat_history_for_retrieval_chain
import os

from api.utils.get_openai_llm import get_llm
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def chat(request):
    try:
        body = json.loads(request.body.decode('utf-8'))
        question = body.get('question')
        namespace = body.get('namespace')
        mode = body.get('mode')
        initial_prompt = body.get('initial_prompt')
        token = body.get('token')
        session_id = body.get('session_id')

        bot = get_object_or_404(Chatbot, token=token)

        if not question:
            return JsonResponse({'error': 'No question in the request'}, status=400)

        sanitized_question = question.strip().replace('\n', ' ')

        vector_store = get_vector_store(StoreOptions(namespace=namespace))
        
        response_text = get_completion_response(vector_store=vector_store, initial_prompt=initial_prompt,mode=mode, sanitized_question=sanitized_question, session_id=session_id)

        ChatHistory.objects.bulk_create([
            ChatHistory(
                id=uuid4(),
                chatbot_id=bot.id,
                from_user=True,
                message=sanitized_question,
                session_id=session_id
            ),
            ChatHistory(
                id=uuid4(),
                chatbot_id=bot.id,
                from_user=False,
                message=response_text,
                session_id=session_id
            )
        ])

        return JsonResponse({'text': response_text})
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Chatbot.DoesNotExist:
        return JsonResponse({'error': 'Chatbot not found'}, status=404)
    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return JsonResponse({'error': 'An error occurred'}, status=500)

def save_chat_to_memory(chat_history, memory):
    for chat in chat_history:
        input_text, output_text = chat
        memory.save_context({"input": input_text}, {"output": output_text})


def get_completion_response(vector_store, mode, initial_prompt, sanitized_question, session_id):
    chain_type = os.getenv("CHAIN_TYPE", "conversation_retrieval")
    chain: QAWithSourcesChain
    if chain_type == 'retrieval_qa':
        chain = getRetrievalQAWithSourcesChain(vector_store, mode, initial_prompt)
        response = chain({"question": sanitized_question}, return_only_outputs=True)
        response_text = response['answer']
    elif chain_type == 'conversation':

        chat_history = get_chat_history_for_retrieval_chain(session_id, limit=2)

        llm = get_llm()
        memory = ConversationBufferMemory(memory_key="chat_history")
        prompt = PromptTemplate.from_template(initial_prompt)
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        save_chat_to_memory(chat_history, memory)

        response = chain({"question": sanitized_question}, return_only_outputs=False)
        print("### response from calling get_completion_response: ", response)
        response_text = response['text']
    elif chain_type == 'conversation_retrieval':
        chain = getConversationRetrievalChain(vector_store, mode, initial_prompt)
        chat_history = get_chat_history_for_retrieval_chain(session_id, limit=5)
        print("### chat_history from calling get_completion_response: ", chat_history)
        response = chain({"question": sanitized_question, "chat_history": chat_history}, return_only_outputs=False)
        print("### response from calling get_completion_response: ", response)
        response_text = response['answer']
    return response_text
