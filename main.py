import os
import streamlit as st
import datetime
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from groq import Groq
from typing import List, Dict
from PIL import Image
import numpy as np
import json

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="LLM Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize directories
DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

CHROMA_DB_PATH = Path("chroma_db")
CHROMA_DB_PATH.mkdir(exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

# Define or get the collection
FLAT_COLLECTION_NAME = "flat_llm_chatbot_collection"
FUNNEL_COLLECTION_NAME = "funnel_llm_chatbot_collection"

# Initialize or get the Flat Collection with Cosine Similarity
if FLAT_COLLECTION_NAME not in [col.name for col in chroma_client.list_collections()]:
    flat_embedding_function = embedding_functions.DefaultEmbeddingFunction()
    flat_collection = chroma_client.create_collection(
        name=FLAT_COLLECTION_NAME,
        embedding_function=flat_embedding_function,
        metadata={"hnsw:space": "cosine"}  # Set distance function to cosine
    )
else:
    flat_collection = chroma_client.get_collection(
        name=FLAT_COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

# Initialize or get the Funnel Collection with L2 Distance
if FUNNEL_COLLECTION_NAME not in [col.name for col in chroma_client.list_collections()]:
    funnel_embedding_function = embedding_functions.DefaultEmbeddingFunction()
    funnel_collection = chroma_client.create_collection(
        name=FUNNEL_COLLECTION_NAME,
        embedding_function=funnel_embedding_function,
        metadata={"hnsw:space": "l2"}  # Set distance function to L2
    )
else:
    funnel_collection = chroma_client.get_collection(
        name=FUNNEL_COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = "llama-3.1-70b-versatile"
if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = "You are a helpful assistant."
if 'flat_rag_search' not in st.session_state:
    st.session_state['flat_rag_search'] = False
if 'funnel_rag_search' not in st.session_state:
    st.session_state['funnel_rag_search'] = False

# Function to save uploaded files with versioning
def save_uploaded_file(uploaded_file) -> Path:
    filename = uploaded_file.name
    name, ext = os.path.splitext(filename)
    today = datetime.datetime.now().strftime("%m_%d_%y")
    version = 1
    while True:
        new_filename = f"{name}_{today}_v{version}{ext}"
        file_path = DOCUMENTS_DIR / new_filename
        if not file_path.exists():
            break
        version += 1
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to vectorize and add documents to both ChromaDB collections
def vectorize_and_add(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # Parse the file as JSON

    def add_documents(node, parent_id=None):
        for key, value in node.items():
            if isinstance(value, dict):
                # Current node is a parent
                doc_id = f"{file_path.stem}_{key}_parent"
                metadata = {
                    "source": file_path.name,
                    "type": "parent",
                    "category": parent_id if parent_id else "root"
                }

                # Add to Flat Collection (individual parent name)
                flat_collection.add(
                    documents=[key],
                    metadatas=[metadata],
                    ids=[doc_id + "_flat"]
                )

                # Add to Funnel Collection (full child object as JSON string)
                full_child_text = json.dumps(value, indent=2)
                funnel_collection.add(
                    documents=[full_child_text],
                    metadatas=[metadata],
                    ids=[doc_id + "_funnel"]
                )

                # Recursively add child nodes
                add_documents(value, parent_id=key)
            elif isinstance(value, list):
                # Handle list of items if necessary
                for idx, item in enumerate(value):
                    doc_id = f"{file_path.stem}_{key}_{idx}"
                    metadata = {
                        "source": file_path.name,
                        "type": "child",
                        "category": parent_id
                    }
                    # Add to Flat Collection
                    flat_collection.add(
                        documents=[item],
                        metadatas=[metadata],
                        ids=[doc_id + "_flat"]
                    )
                    # Add to Funnel Collection
                    funnel_collection.add(
                        documents=[item],
                        metadatas=[metadata],
                        ids=[doc_id + "_funnel"]
                    )
            else:
                # Current node is a leaf node
                doc_id = f"{file_path.stem}_{key}_leaf"
                metadata = {
                    "source": file_path.name,
                    "type": "leaf",
                    "category": parent_id
                }
                # Add to Flat Collection
                flat_collection.add(
                    documents=[f"{key}: {value}"],
                    metadatas=[metadata],
                    ids=[doc_id + "_flat"]
                )
                # Add to Funnel Collection
                funnel_collection.add(
                    documents=[f"{key}: {value}"],
                    metadatas=[metadata],
                    ids=[doc_id + "_funnel"]
                )

    # Start adding documents from the root of the JSON
    add_documents(data)


# Flat RAG Search with Cosine Similarity
def flat_rag_search(query: str, n_results: int = 3) -> List[Dict]:
    results = flat_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results['documents'][0], results['metadatas'][0], results['distances'][0]

# Funnel RAG Search with L2 Distance
def funnel_rag_search(query: str, n_results: int = 3) -> List[Dict]:
    # Step 1: Search parent-level embeddings
    parent_results = funnel_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    # Get relevant parent documents
    parent_docs = parent_results['documents'][0]
    parent_metadatas = parent_results['metadatas'][0]
    parent_distances = parent_results['distances'][0]
    filtered_children = []
    child_metadatas = []
    child_distances = []

    # Step 2: Retrieve full child objects based on the parent results
    for parent_doc, meta, distance in zip(parent_docs, parent_metadatas, parent_distances):
        category = meta.get('category', None)
        if category:
            # Since parent_doc now contains the full child object as JSON string
            filtered_children.append(parent_doc)  # Add the full child object
            child_metadatas.append(meta)
            child_distances.append(distance)

    return filtered_children, child_metadatas, child_distances


# Initialize Groq client
def initialize_groq_client() -> Groq:
    api_key = st.session_state.get('api_key', "")
    if not api_key:
        st.warning("Please enter your Groq API Key in the Settings page.")
        return None
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

# Handle user queries
@st.fragment
def handle_query(query: str):
    if not query.strip():
        st.warning("Please enter a valid query.")
        return

    flat_search = st.session_state.get('flat_rag_search', False)
    funnel_search = st.session_state.get('funnel_rag_search', False)

    if flat_search and funnel_search:
        st.error("Only one RAG Search method can be active at a time.")
        return

    groq_client = initialize_groq_client()
    if not groq_client:
        return

    if flat_search:
        docs, metadatas, distances = flat_rag_search(query)
        search_type = "Flat RAG Search"
        context = "\n".join(docs)
        messages = [
            {"role": "system", "content": st.session_state['system_prompt']},
            {"role": "user", "content": query + f"\nContext:\n{context}"}
        ]
    elif funnel_search:
        docs, metadatas, distances = funnel_rag_search(query)
        search_type = "Funnel RAG Search"

        # Assuming each doc in docs is a JSON string representing the full child object
        formatted_context = []
        for doc in docs:
            try:
                json_obj = json.loads(doc)
                formatted_context.append(json.dumps(json_obj, indent=2))
            except json.JSONDecodeError:
                formatted_context.append(doc)  # If not JSON, add as-is

        context = "\n".join(formatted_context)
        messages = [
            {"role": "system", "content": st.session_state['system_prompt']},
            {"role": "user", "content": query + f"\nContext:\n{context}"}
        ]
    else:
        # No RAG search, send user prompt as-is
        messages = [
            {"role": "system", "content": st.session_state['system_prompt']},
            {"role": "user", "content": query}
        ]

    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model=st.session_state['model_name']
        )
        answer = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return

    # Append user and assistant messages to chat history
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        st.markdown(answer)

    if flat_search or funnel_search:
        with st.expander("Top 3 Sources"):
            data = {
                "Weight": [f"{1 - distance:.4f}" for distance in distances[:3]],  # Adjust based on your similarity metric
                "File Name": [meta['source'] for meta in metadatas[:3]],
                "Extracted Text": docs[:3]
            }
            df = pd.DataFrame(data)
            st.dataframe(df)


# Clear chat history
def clear_chat_history():
    st.session_state['chat_history'] = []
    st.toast("Chat history cleared.")

# Chat Page
def chat_page():
    st.title("Chat")

    # Sidebar Toggles
    st.sidebar.header("RAG Search Options")

    def toggle_flat():
        st.session_state['flat_rag_search'] = not st.session_state['flat_rag_search']
        if st.session_state['flat_rag_search']:
            st.session_state['funnel_rag_search'] = False

    def toggle_funnel():
        st.session_state['funnel_rag_search'] = not st.session_state['funnel_rag_search']
        if st.session_state['funnel_rag_search']:
            st.session_state['flat_rag_search'] = False

    st.sidebar.toggle("Flat RAG Search", value=st.session_state['flat_rag_search'], on_change=toggle_flat)
    st.sidebar.toggle("Funnel RAG Search", value=st.session_state['funnel_rag_search'], on_change=toggle_funnel)

    # Chat History and Input
    for chat in st.session_state['chat_history']:
        if chat['role'] == "user":
            st.chat_message("user").write(chat['content'])
        elif chat['role'] == "assistant":
            st.chat_message("assistant").write(chat['content'])

    user_input = st.chat_input("Enter your message here...")
    if user_input:
        handle_query(user_input)

# Settings Page
def settings_page():
    st.title("Settings")
    st.header("LLM Config")

    model_name = st.text_input("Model Name", value=st.session_state['model_name'])
    st.session_state['model_name'] = model_name

    api_key = st.text_input("Groq API Key", type="password", value=st.session_state['api_key'])
    st.session_state['api_key'] = api_key

    system_prompt = st.text_area("System Prompt", value=st.session_state['system_prompt'], height=100)
    st.session_state['system_prompt'] = system_prompt

    if st.button("Clear Chat History"):
        clear_chat_history()

# Documents Page
def documents_page():
    st.title("Documents")

    uploaded_files = st.file_uploader("Upload Documents", type=["txt", "csv",'png', 'jpg'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            vectorize_and_add(file_path)
        st.toast(f"Uploaded and processed {len(uploaded_files)} file(s).")

    documents = sorted(DOCUMENTS_DIR.glob("*.*"))
    doc_names = [doc.name for doc in documents]
    if doc_names:
        selected_doc = st.selectbox("Select a Document to Preview", options=doc_names)
        if selected_doc:
            doc_path = DOCUMENTS_DIR / selected_doc
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.text_area("Document Preview", value=content, height=300)
    else:
        st.info("No Files Uploaded Yet.")

    if st.button("Clear Documents from Folder and Vector Store"):
        # Delete vector store collection
        chroma_client.delete_collection(name=FLAT_COLLECTION_NAME)
        chroma_client.delete_collection(name=FUNNEL_COLLECTION_NAME)
        
        # Delete all files from the local documents folder
        for file in DOCUMENTS_DIR.glob("*"):
            file.unlink()  # Deletes the file
        st.toast("All documents have been deleted from the vector store.")
        st.rerun()

# Main Navigation
def main():
    # Page Navigation
    pages = [
        st.Page(chat_page, title="Chat", default=True, icon="🤖"),
        st.Page(settings_page, title="Settings", icon="⚙️"),
        st.Page(documents_page, title="Documents", icon="📄"),
    ]
    
    
    page = st.navigation(pages)
    page.run()

if __name__ == "__main__":
    main()
