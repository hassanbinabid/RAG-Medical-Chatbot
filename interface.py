import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="ClinicaBot 🏥",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 ClinicaBot")
st.caption("Medical Chatbot based on Top Medicine Encyclopedias")
st.divider()

# ===============================
# Load LLM & FAISS (Cached)
# ===============================

@st.cache_resource
def load_llm():
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found! Make sure it's set in your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.5,
        google_api_key=GEMINI_API_KEY
    )

@st.cache_resource
def load_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

@st.cache_resource
def build_rag_chain():
    llm = load_llm()
    db = load_vectorstore()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are ClinicaBot, a helpful and knowledgeable medical assistant. 
         Use the following retrieved context from top medicine encyclopedias to 
         answer the user's question accurately and clearly.
         If you don't know the answer or it's not in the context, say so clearly.
         Do not make up medical information.
         
         Context: {context}"""),
        ("human", "{input}")
    ])

    retriever = db.as_retriever(search_kwargs={'k': 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ===============================
# Initialize Chat History
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# Display Chat History
# ===============================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===============================
# Handle User Input
# ===============================

if user_query := st.chat_input("Ask ClinicaBot anything..."):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical encyclopedias..."):
            try:
                rag_chain = build_rag_chain()
                response_placeholder = st.empty()
                full_response = ""

                for chunk in rag_chain.stream(user_query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ===============================
# Sidebar
# ===============================

with st.sidebar:
    st.header("🏥 ClinicaBot")
    st.write("ClinicaBot uses RAG (Retrieval-Augmented Generation) to answer medical questions based on top medicine encyclopedias.")
    st.divider()
    st.subheader("📚 How it works")
    st.write("1. Your question is matched against the medical encyclopedias")
    st.write("2. Relevant passages are retrieved from FAISS vector store")
    st.write("3. Gemini LLM generates an accurate answer based on the context")
    st.divider()
    st.warning("⚠️ This is not a substitute for professional medical advice. Always consult a qualified doctor.")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()