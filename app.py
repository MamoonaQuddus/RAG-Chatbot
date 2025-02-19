import os
import warnings
import logging
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# ✅ Set up Streamlit page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ---- Disable warnings and logs ----
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---- Load External CSS ----
def load_css():
    """Reads and applies styles from styles.css"""
    css_file = "styles.css"
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()  # Apply styles

# ---- Sidebar UI ----
st.sidebar.title("⚙️ Chatbot Settings")
st.sidebar.info("Upload a **PDF, DOCX, or TXT** file and ask anything related to it!")

# ---- API Key Input ----
st.sidebar.subheader("🔑 Enter Your Groq API Key:")
groq_api_key = st.sidebar.text_input("API Key", type="password")

if groq_api_key:
    st.sidebar.success("✅ API Key Saved Successfully")
else:
    st.sidebar.error("❌ Please enter your API Key.")

st.title("🤖 AI Chatbot")

# ---- File Upload Section ----
uploaded_file = st.file_uploader("📂 Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# ---- Session State Initialization ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Process Uploaded File ----
@st.cache_resource
def get_vectorstore(file):
    """Processes uploaded document and creates a vector store."""
    try:
        if not file:
            return None

        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            return None

        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ).from_loaders([loader])

        return index.vectorstore
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

vectorstore = None
if uploaded_file:
    with st.spinner("📚 Processing file..."):
        vectorstore = get_vectorstore(uploaded_file)

# ---- Chat Input Section ----
st.subheader("💬 Ask Your Question:")
prompt = st.chat_input("Type your question here...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat UI
    with st.chat_message("user"):
        st.write(prompt)

    # ---- AI Model ----
    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are an expert AI chatbot. Provide the most accurate, detailed, and to-the-point response.
        Avoid unnecessary small talk. Answer the following question: {user_prompt}.
    """)

    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # ---- Fetch AI Response ----
    try:
        with st.spinner("🤖 Generating response..."):
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"]

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display AI response in chat UI
        with st.chat_message("assistant"):
            st.write(response)

        # Show Sources
        if "source_documents" in result and result["source_documents"]:
            st.sidebar.subheader("📌 Sources Used:")
            for doc in result["source_documents"]:
                st.sidebar.text(f"- {doc.metadata['source']}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.markdown(f'<div class="error-box">⚠️ An error occurred: {str(e)}</div>', unsafe_allow_html=True)
