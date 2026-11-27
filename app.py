import os
import re
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SUPPORTED_EXTENSIONS = [
    "pdf",
    "docx",
    "txt",
    "md",
    "csv",
]

DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# --------------------------------------------------
# API KEY VALIDATION
# --------------------------------------------------

def is_valid_api_key(api_key):
    if not api_key:
        return False

    return api_key.startswith("sk-or-")


# --------------------------------------------------
# BUILD LANGUAGE MODEL
# --------------------------------------------------

def build_llm(api_key, model_name):

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        streaming=True,
    )


# --------------------------------------------------
# BUILD EMBEDDINGS
# --------------------------------------------------

def build_embeddings(api_key, embedding_model):

    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


# --------------------------------------------------
# DOCUMENT HASH
# --------------------------------------------------

def get_file_hash(file_bytes):

    return hashlib.sha256(file_bytes).hexdigest()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

def initialize_session_state():

    defaults = {
        "vector_store": None,
        "chat_history": [],
        "indexed_files": [],
        "processed_hashes": [],
        "all_documents": [],
        "extracted_questions": [],
        "embedding_model": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value
# --------------------------------------------------
# LOAD UPLOADED DOCUMENT
# --------------------------------------------------

def load_uploaded_file(uploaded_file):

    documents = []

    file_name = uploaded_file.name
    extension = Path(file_name).suffix.lower()

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        return documents

    with tempfile.TemporaryDirectory() as temp_dir:

        file_path = Path(temp_dir) / file_name

        file_path.write_bytes(file_bytes)

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))

        elif extension in [".txt", ".md"]:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        elif extension == ".csv":
            loader = CSVLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        documents = loader.load()

    for document in documents:

        document.metadata["source_file"] = file_name

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]


# --------------------------------------------------
# EXTRACT NUMBERED QUESTIONS
# --------------------------------------------------

def parse_questions_from_documents(documents):

    questions = []

    question_pattern = re.compile(
        r"^\s*(?:Q(?:uestion)?\s*)?(\d+)[.)]\s*(.+)",
        re.IGNORECASE
    )

    for document in documents:

        lines = document.page_content.splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            match = question_pattern.match(line)

            if match:

                number = match.group(1)
                question_text = match.group(2).strip()

                if question_text:

                    questions.append({
                        "number": number,
                        "text": question_text,
                        "source": document.metadata.get(
                            "source_file",
                            "Unknown"
                        ),
                        "page": document.metadata.get(
                            "page",
                            "Unknown"
                        ),
                    })

    return questions


# --------------------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(
    documents,
    chunk_size,
    chunk_overlap
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# INDEX DOCUMENTS
# --------------------------------------------------

def index_documents(
    uploaded_files,
    api_key,
    embedding_model,
    chunk_size,
    chunk_overlap
):

    if not uploaded_files:
        raise ValueError("Please upload documents first.")

    embeddings = build_embeddings(
        api_key,
        embedding_model
    )

    all_new_documents = []
    skipped_files = []

    existing_hashes = set(
        st.session_state.processed_hashes
    )

    new_hashes = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = get_file_hash(file_bytes)

        if file_hash in existing_hashes:
            continue

        try:

            documents = load_uploaded_file(
                uploaded_file
            )

            if not documents:

                skipped_files.append(uploaded_file.name)
                continue

            all_new_documents.extend(documents)

            new_hashes.append(file_hash)

        except Exception as error:

            st.warning(
                f"Could not load {uploaded_file.name}: {error}"
            )

    if not all_new_documents:

        return 0, skipped_files

    chunks = split_documents(
        all_new_documents,
        chunk_size,
        chunk_overlap
    )

    if not chunks:

        return 0, skipped_files

    # Create a new vector store when needed.
    if (
        st.session_state.vector_store is None
        or st.session_state.embedding_model != embedding_model
    ):

        st.session_state.vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.session_state.embedding_model = embedding_model

    else:

        st.session_state.vector_store.add_documents(
            chunks
        )

    # Save indexed information.
    st.session_state.all_documents.extend(
        all_new_documents
    )

    st.session_state.extracted_questions.extend(
        parse_questions_from_documents(
            all_new_documents
        )
    )

    st.session_state.processed_hashes.extend(
        new_hashes
    )

    for document in all_new_documents:

        source = document.metadata.get(
            "source_file",
            "Unknown"
        )

        if source not in st.session_state.indexed_files:

            st.session_state.indexed_files.append(
                source
            )

    return len(chunks), skipped_files
# --------------------------------------------------
# CONVERT CHAT HISTORY
# --------------------------------------------------

def to_langchain_history(messages, max_turns=6):

    history = []

    recent_messages = messages[-(max_turns * 2):]

    for message in recent_messages:

        if message["role"] == "user":

            history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    return history


# --------------------------------------------------
# CONTEXTUALIZE USER QUESTION
# --------------------------------------------------

def contextualize_query(
    llm,
    question,
    chat_history
):

    if not chat_history:
        return question

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            Given the conversation history and the latest
            user question, rewrite the latest question so
            that it can be understood independently.

            Do not answer the question.
            Return only the rewritten question.
            """
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )

    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({

        "chat_history": chat_history,
        "question": question

    })


# --------------------------------------------------
# FORMAT RETRIEVED DOCUMENTS
# --------------------------------------------------

def format_context(documents):

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source_file",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(

            f"""
            Source {index}
            File: {source}
            Page: {page}

            Content:
            {document.page_content}
            """

        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# GENERATE RAG ANSWER
# --------------------------------------------------

def stream_rag_answer(
    llm,
    question,
    context,
    chat_history
):

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are a helpful document question-answering
            assistant.

            Answer the user's question using the supplied
            document context.

            Instructions:

            1. Give clear and simple answers.
            2. Use the provided context as your main source.
            3. Do not invent information.
            4. If the answer is not available in the documents,
               clearly say that it was not found.
            5. Use headings and bullet points when useful.
            6. Keep the answer relevant to the question.

            DOCUMENT CONTEXT:

            {context}
            """
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )

    ])

    chain = prompt | llm | StrOutputParser()

    return chain.stream({

        "context": context,
        "chat_history": chat_history,
        "question": question

    })


# --------------------------------------------------
# FIND A NUMBERED QUESTION
# --------------------------------------------------

def answer_document_question(question):

    match = re.search(
        r"(?:question|q)\s*(\d+)",
        question,
        re.IGNORECASE
    )

    if not match:
        return None

    requested_number = match.group(1)

    matching_questions = [

        item
        for item in st.session_state.extracted_questions

        if item["number"] == requested_number

    ]

    if not matching_questions:
        return None

    results = []

    for item in matching_questions:

        results.append(

            f"""
            **Question {item['number']}**

            {item['text']}

            Source: {item['source']}
            Page: {item['page']}
            """

        )

    return "\n\n".join(results)


# --------------------------------------------------
# RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_documents(
    question,
    api_key,
    embedding_model,
    k=4
):

    vector_store = st.session_state.vector_store

    if vector_store is None:
        return []

    embeddings = build_embeddings(
        api_key,
        embedding_model
    )

    # Use the existing vector store's similarity search.
    return vector_store.similarity_search(
        question,
        k=k
    )
# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main():

    st.set_page_config(
        page_title="RAG Document Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    st.title("🤖 RAG Document Chatbot")

    st.caption(
        "Upload documents and ask questions based on their content."
    )

    # --------------------------------------------------
    # SIDEBAR SETTINGS
    # --------------------------------------------------

    with st.sidebar:

        st.header("⚙️ Settings")

        try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENROUTER_API_KEY", "")

        chat_model = st.selectbox(
            "Chat Model",
            options=[
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-haiku",
                "google/gemini-2.0-flash-001"
            ]
        )

        embedding_model = DEFAULT_EMBEDDING_MODEL

        st.text_input(
            "Embedding Model",
            value=embedding_model,
            disabled=True
        )

        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50
        )

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []
            st.rerun()

        if st.button(
            "🔄 Reset Documents",
            use_container_width=True
        ):

            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.indexed_files = []
            st.session_state.processed_hashes = []
            st.session_state.all_documents = []
            st.session_state.extracted_questions = []
            st.session_state.embedding_model = None

            st.success("Documents reset successfully.")
            st.rerun()

    # --------------------------------------------------
    # DOCUMENT UPLOAD
    # --------------------------------------------------

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True
    )

    # --------------------------------------------------
    # PROCESS AND INDEX DOCUMENTS
    # --------------------------------------------------

    if st.button(
        "🔍 Process and Index Documents",
        use_container_width=True
    ):

        if not is_valid_api_key(api_key):

            st.error(
                "Please enter a valid OpenRouter API key."
            )

        elif not uploaded_files:

            st.warning(
                "Please upload documents first."
            )

        else:

            try:

                with st.spinner(
                    "Processing and indexing documents..."
                ):

                    chunk_count, skipped_files = index_documents(
                        uploaded_files=uploaded_files,
                        api_key=api_key,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                if chunk_count > 0:

                    st.success(
                        "Documents processed successfully!"
                    )

                else:

                    st.info(
                        "No new document chunks were indexed."
                    )

                st.write(
                    f"**New chunks indexed:** {chunk_count}"
                )

                st.write(
                    f"**Documents indexed:** "
                    f"{len(st.session_state.indexed_files)}"
                )

                st.write(
                    f"**Numbered questions identified:** "
                    f"{len(st.session_state.extracted_questions)}"
                )

                if skipped_files:

                    st.warning(
                        "No readable text found in: "
                        + ", ".join(skipped_files)
                    )

            except Exception as error:

                st.error(
                    f"Error while processing documents: {error}"
                )

    # --------------------------------------------------
    # SHOW INDEXED DOCUMENTS
    # --------------------------------------------------

    if st.session_state.indexed_files:

        with st.expander("📚 Indexed Documents"):

            for file_name in st.session_state.indexed_files:

                st.write(f"📄 {file_name}")

    # --------------------------------------------------
    # CHAT INTERFACE
    # --------------------------------------------------

    st.divider()

    st.header("💬 Chat with Your Documents")

    # Display previous messages.

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("📖 Sources"):

                    for source in message["sources"]:

                        st.write(source)

    # --------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    if question:

        if st.session_state.vector_store is None:

            st.warning(
                "Please upload and index documents before asking questions."
            )

        elif not is_valid_api_key(api_key):

            st.error(
                "Please enter your OpenRouter API key."
            )

        else:

            # Store previous conversation before adding
            # the current question.

            chat_history = to_langchain_history(
                st.session_state.chat_history
            )

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):

                st.markdown(question)

            with st.chat_message("assistant"):

                try:

                    # Check for a numbered question.

                    numbered_question = answer_document_question(
                        question
                    )

                    if numbered_question:

                        answer = numbered_question

                        st.markdown(answer)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                    else:

                        # Create language model.

                        llm = build_llm(
                            api_key,
                            chat_model
                        )

                        # Rewrite the question using chat history.

                        with st.spinner(
                            "Understanding your question..."
                        ):

                            rewritten_question = contextualize_query(
                                llm,
                                question,
                                chat_history
                            )

                        # Retrieve relevant documents.

                        with st.spinner(
                            "Searching your documents..."
                        ):

                            retrieved_docs = retrieve_documents(
                                rewritten_question,
                                api_key,
                                embedding_model,
                                k=4
                            )

                        if not retrieved_docs:

                            answer = (
                                "I could not find relevant information "
                                "in the uploaded documents."
                            )

                            st.markdown(answer)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                        else:

                            # Prepare document context.

                            context = format_context(
                                retrieved_docs
                            )

                            # Generate the answer only once.
                            # Streamlit collects the streamed text
                            # and returns the complete answer.

                            answer = st.write_stream(
                                stream_rag_answer(
                                    llm,
                                    question,
                                    context,
                                    chat_history
                                )
                            )

                            # Collect source information.

                            sources = []

                            for document in retrieved_docs:

                                source_name = document.metadata.get(
                                    "source_file",
                                    "Unknown"
                                )

                                page_number = document.metadata.get(
                                    "page",
                                    "Unknown"
                                )

                                source_text = (
                                    f"{source_name} — Page {page_number}"
                                )

                                if source_text not in sources:

                                    sources.append(source_text)

                            # Display sources.

                            if sources:

                                st.markdown("**📖 Sources**")

                                for source in sources:

                                    st.write(f"- {source}")

                            # Save answer and sources.

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })

                except Exception as error:

                    st.error(
                        f"Error generating answer: {error}"
                    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
