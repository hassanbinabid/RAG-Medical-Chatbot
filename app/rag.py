import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ===============================
# Load LLM
# ===============================

def load_llm():
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found! Make sure it's set in your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.5,
        google_api_key=GEMINI_API_KEY
    )

# ===============================
# Load Vectorstore
# ===============================

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

# ===============================
# Build RAG Chain
# ===============================

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
# Get Answer (Main Function)
# ===============================

# Build chain once at module level so it's reused across requests
rag_chain = build_rag_chain()

def get_answer(question: str) -> str:
    response = rag_chain.invoke(question)
    return response