from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

from textwrap import dedent


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
    
    cidentify_description = "You are an Assistant called 'Statement of Purpose(SOP) Writing Assistant' that writes a SOP from given template and data."
    cidentify_instructions = [
            "Answer the users question by only using `search_knowledge_base` tool.",
            "Write a statement of purpose for an undergraduate level.",
            "Answer all questions according to given document which includes student's data.",
            "Answer the questions directly. Don't include the tool name you used in your answer.",
            "NEVER mention your knowledge base or say 'According to the search_knowledge_base tool' or 'According to {some_tool} tool'.",
            "Respond directly to the user with your answer, do not say 'here is the answer' or 'this is the answer' or 'According to the information provided' or 'I found ...'",
            "Voice and Tone: Write in the active voice and maintain a positive, formal, yet conversational tone.",
            "Continuity: Ensure the SOP has a logical flow and coherence.",
            "Use Examples: Include specific examples from your life to support your statements.",
            "Tailoring: Customize your SOP for each institution or course to which you apply. Get institution name from student's data file.",
        ]
    
    cidentify_format = """
        <sop_format>
        ## [Course Name] Application Statement of Purpose

        ### Introduction:
        I am set to complete my high school program in [high school course name] in a few months’ time, and I realize that this is a crucial stage in my career where I must decide the direction for a successful future. My profound interest in [subject] is the primary reason I seek to pursue a graduate degree in the same field. Furthermore, the rapid advancements in [subject area] recently have made the role of a [job name] increasingly indispensable. To achieve my goal of becoming a successful [job role], it is essential that I study at [university name], which offers a rewarding research program, excellent facilities, and an inspiring environment.

        ### Academic Background:
        I have always shown a keen interest in [subjects], which has significantly enhanced my [skills]. These strengths have helped me achieve [score]% in my [last study level]. I scored [score]% in senior secondary school, with [score]% in [relevant subject] and [score]% in [relevant subject]. My overall percentage in [subjects] is [score]%.

        ### Competitive Examinations:
        Given my passion for [field of interest], pursuing higher studies in [subject] was a natural choice. I demonstrated my capabilities through [competitive examinations], passing with flying colors and ranking among the top X% of all candidates. I then enrolled in [course] at [school name], affiliated with the prestigious [school name] in [locality]. I chose [course] for my high school major due to its potential to help me realize my goals and contribute to society. Throughout my high school program, I have maintained an excellent academic record.

        ### high school Experience:
        My high school program has provided me with comprehensive exposure to various courses that I found fascinating, such as [topics covered as part of the program]. I strongly believe in practical learning, and the hope of discovering groundbreaking results through experimentation is very appealing to me. Beyond the classroom, I have engaged in numerous industrial visits, gaining a closer look at the practical applications of my studies. I presented [number/topics of paper presentations] at national and state levels and organized several technical events in college. Elected as the president of my department, I demonstrated leadership qualities, communication skills, and high performance. I also actively participated in [relevant topic] competitions organized by my university.

        ### Projects:
        During my final year, I undertook two significant projects – [project names]. As the [your role in the project], I learned to overcome numerous practical challenges with limited resources. Both projects were successfully completed on time, under the guidance of technical experts from my school.

        ### Career Objectives:
        My objective in pursuing an MS in [subject] is to acquire in-depth knowledge and hone my intellectual abilities. Ten years from now, I envision myself working in the [industry name] sector, applying my learning to contribute to its development.

        ### Conclusion:
        I believe that [university name], with its world-renowned high-tech facilities, is the ideal place for my graduate studies. I eagerly anticipate your acceptance of my application for the [program name] program.

        </sop_format>
        """

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
        description=cidentify_description,             
        instructions=cidentify_instructions,
        add_to_system_prompt=dedent(cidentify_format),
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
