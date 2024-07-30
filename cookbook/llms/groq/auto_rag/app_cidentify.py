import nest_asyncio
import os
from typing import List

import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from assistant import get_auto_rag_assistant  # type: ignore

nest_asyncio.apply()
st.set_page_config(
    page_title="Autonomous RAG",
    page_icon=":orange_heart:",
)
st.title("Autonomous RAG with Llama3")
st.markdown("##### :orange_heart: built using [phidata](https://github.com/phidatahq/phidata)")

def chat_token_size(chat_history:list) -> int:
    n = 0
    for msg in chat_history:
        n += len(msg['content'].strip().split())
    return n
def restart_assistant():
    logger.debug("---*--- Restarting Assistant ---*---")
    st.session_state["auto_rag_assistant"] = None
    st.session_state["auto_rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


def main() -> None:
    # Get LLM model
    llm_model = st.sidebar.selectbox("Select LLM", options=["llama3-70b-8192", "llama3-8b-8192"])
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    # Get Embeddings model
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["text-embedding-3-small", "nomic-embed-text"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )
    # Set assistant_type in session state
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    # Get the assistant
    auto_rag_assistant: Assistant
    if "auto_rag_assistant" not in st.session_state or st.session_state["auto_rag_assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        auto_rag_assistant = get_auto_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["auto_rag_assistant"] = auto_rag_assistant
    else:
        auto_rag_assistant = st.session_state["auto_rag_assistant"]

    # Create assistant run (i.e. log to database) and save run_id in session state
    try:
        st.session_state["auto_rag_assistant_run_id"] = auto_rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    # Load existing messages
    assistant_chat_history = auto_rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        logger.debug(f"Chat Length:{chat_token_size(assistant_chat_history)}")
        # logger.debug(str(assistant_chat_history))

        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        # st.session_state["messages"] = [{"role": "assistant", "content": "Upload a doc and ask me questions..."}]
        st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to ETIO Services... How can I help you?"}]

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            resp_container = st.empty()
            # Streaming is not supported with function calling on Groq atm
            response = auto_rag_assistant.run(question, stream=False)
            logger.debug(f"Question:{question}, response:{response}")

            resp_container.markdown(response)  # type: ignore
            # Once streaming is supported, the following code can be used
            # response = ""
            # for delta in auto_rag_assistant.run(question):
            #     response += delta  # type: ignore
            #     resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # Load knowledge base
    if auto_rag_assistant.knowledge_base:
        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        auto_rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()
                restart_assistant()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 1000

        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            # st.write("upload:", uploaded_file)
            alert = st.sidebar.info("Processing PDF...", icon="🧠")
            rag_name = uploaded_file.name.split(".")[0]
            if f"{rag_name}_uploaded" not in st.session_state:
                reader = PDFReader()
                rag_documents: List[Document] = reader.read(uploaded_file)
                if rag_documents:
                    auto_rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                else:
                    st.sidebar.error("Could not read PDF")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()
            restart_assistant()
        
        ## ADD a Collection of files
        if "folder_uploader_key" not in st.session_state:
            st.session_state["folder_uploader_key"] = 101   # ?? a unique key?

        #I:\My Drive\LLM_Solutions\Autonomous_RAG\data
        input_folder = st.sidebar.text_input(
                "Add Folder to Knowledge Base", type="default", key=st.session_state["folder_uploader_key"]
            )

        if input_folder is not None:
            # logger.debug(input_folder)
            # Scan the folder with files.
            file_names = []
            if os.path.isdir(input_folder):
                included_extensions = ['pdf']
                file_names = [fn for fn in os.listdir(input_folder)
                            if any(fn.endswith(ext) for ext in included_extensions)]
            
            else:
                logger.debug("Not a folder path..")

        loadAll_btn = st.sidebar.button("Add all files!..")
        if loadAll_btn:
            if file_names is not None:
                # loadAll_btn.disable()
                logger.debug("Loading files...")
                logger.debug(file_names)
                dir = input_folder
                for file_name in file_names:
                    logger.debug(file_name)
                    alert = st.sidebar.info("Processing : " + file_name, icon="🧠")
                    rag_name = file_name.split(".")[0]
                    file = dir + "/" + file_name
                    
                    if f"{rag_name}_uploaded" not in st.session_state:
                        try:
                            reader = PDFReader()
                            rag_documents: List[Document] = reader.read(file)
                        except Exception as e:
                            logger.error(e.__traceback__)
                            continue

                        if rag_documents:
                            auto_rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                        else:
                            st.sidebar.error("Could not read PDF:" + file_name)
                        st.session_state[f"{rag_name}_uploaded"] = True
                    alert.empty()
                    if "file_uploader_key" in st.session_state:
                        st.session_state["file_uploader_key"] += 1
                st.rerun()
                restart_assistant()
            else:
                st.sidebar.error("Enter a folder path first..")

    if auto_rag_assistant.knowledge_base and auto_rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            auto_rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")
            restart_assistant()

    if auto_rag_assistant.storage:
        auto_rag_assistant_run_ids: List[str] = auto_rag_assistant.storage.get_all_run_ids()
        new_auto_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=auto_rag_assistant_run_ids)
        if st.session_state["auto_rag_assistant_run_id"] != new_auto_rag_assistant_run_id:
            logger.info(f"---*--- Loading {llm_model} run: {new_auto_rag_assistant_run_id} ---*---")
            st.session_state["auto_rag_assistant"] = get_auto_rag_assistant(
                llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_auto_rag_assistant_run_id
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()

    if "embeddings_model_updated" in st.session_state:
        st.sidebar.info("Please add documents again as the embeddings model has changed.")
        st.session_state["embeddings_model_updated"] = False


main()

# import os

# p = os.path.abspath(os.getcwd())
# st.write("Path:", p)