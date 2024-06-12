from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


def get_auto_rag_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Auto RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    embeddings_table = (
        "auto_rag_documents_groq_ollama" if embeddings_model == "nomic-embed-text" else "auto_rag_documents_groq_openai"
    )

    default_description = "You are an Assistant called 'AutoRAG' that answers questions by calling functions."
    default_instructions = [
            "First get additional information about the users question.",
            "You can either use the `search_knowledge_base` tool to search your knowledge base or the `duckduckgo_search` tool to search the internet.",
            "If the user asks about current events, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation, use the `get_chat_history` tool to get your chat history with the user.",
            "Carefully process the information you have gathered and provide a clear and concise answer to the user.",
            "Respond directly to the user with your answer, do not say 'here is the answer' or 'this is the answer' or 'According to the information provided' or 'I found ...'",
            "Don't include the tool name you used in your answer."
            "NEVER mention your knowledge base or say 'According to the search_knowledge_base tool' or 'According to {some_tool} tool'.",
            "Show your reference document in short APA format.",
            "Show the page number of your reference like (p33).",
        ]
    
    etio_description = "You are an Assistant called 'ETIO Chatbot' that answers questions by calling functions."
    etio_instructions = [
            "Respond to greetings with a nice greeting.",
            "Give as possible as short answers.",
            "Answer all questions according to ETIO Consulting Services. Do not answer unrelated questions.",
            "When the question is general like no relation to ETIO or no mentions about ETIO, do not give the answer to that question and say 'I can answer your specific questions about ETIO services.'.",
            "Answer the questions directly. Don't include the tool name you used in your answer.",
            "NEVER mention your knowledge base or say 'According to the search_knowledge_base tool' or 'According to {some_tool} tool'.",
            "When the user ask about you, say 'I am a robot to help you about ETIO services.'",
            "Answer the users question by only using `search_knowledge_base` tool.",
            "If the user asks to summarize the conversation, use the `get_chat_history` tool to get your chat history with the user.",
            # "Carefully process the information you have gathered and provide a clear and concise answer to the user.",
            "Respond directly to the user with your answer, do not say 'here is the answer' or 'this is the answer' or 'According to the information provided' or 'I found ...'",
            "Show your reference in short APA format.",
        ]

    return Assistant(
        name="auto_rag_assistant_groq",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="auto_rag_assistant_groq", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 3 references are added to the prompt
            num_documents=3,
        ),
        description=etio_description,               ## ??
        instructions=etio_instructions,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search for information
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        tools=[DuckDuckGo()],
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # Adds chat history to messages
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
