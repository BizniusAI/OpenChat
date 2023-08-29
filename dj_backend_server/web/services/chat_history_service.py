from typing import List, Optional, Tuple
from web.models.chat_histories import ChatHistory

def get_chat_history_for_retrieval_chain(session_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Fetches limited ChatHistory entries by session ID and converts to chat_history format.

    Args:
        session_id (str): The session ID to fetch chat history for 
        limit (int, optional): Maximum number of entries to retrieve

    Returns:
        list[tuple[str, str]]: List of tuples of (user_query, bot_response) 
    """
    
    # Query and limit results if a limit is provided
    query = ChatHistory.objects.filter(session_id=session_id).order_by('created_at')

    print("Chat history query: ", query)

    # if limit:
    #     query = query[:limit]
    if not query:
        return []

    chat_history = []
    pending_user_query = None

    for entry in query:
        print(f"### Entry from user: {entry.from_user}, Message: {entry.message}")
        from_user = entry.from_user == "True"  # Convert the string to a boolean
        if from_user:
            pending_user_query = entry.message
            print("### pending user query: ", pending_user_query)
        else:
            bot_response = entry.message or "No response from bot"
            print("### bot response: ", bot_response)
            if pending_user_query:
                chat_history.append((pending_user_query, bot_response))
                print("### newly added tuple, and set pending_user_query to none: ", (pending_user_query, bot_response))
                pending_user_query = None

    # Handle any remaining unmatched user query
    if pending_user_query:
        chat_history.append((pending_user_query, "No response from bot"))
        print("### newly added tuple: ", (pending_user_query, "No response from bot"))

    return chat_history[-limit:] if limit else chat_history