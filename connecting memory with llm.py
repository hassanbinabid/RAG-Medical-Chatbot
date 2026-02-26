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
# Step 1: Setup Gemini LLM
# ===============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found! Make sure it's set in your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    google_api_key=GEMINI_API_KEY
)

# ===============================
# Step 2: Load FAISS Database
# ===============================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# ===============================
# Step 3: Build RAG Chain
# ===============================

retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a helpful medical assistant. Use the following retrieved context 
     from the Gale Encyclopedia of Medicine to answer the user's question accurately.
     If you don't know the answer or it's not in the context, say so clearly.
     Do not make up medical information.
     
     Context: {context}"""),
    ("human", "{input}")
])

retriever = db.as_retriever(search_kwargs={'k': 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | retrieval_qa_chat_prompt
    | llm
    | StrOutputParser()
)

# ===============================
# Step 4: Ask Question
# ===============================

user_query = input("Write Query Here: ")

response = rag_chain.invoke(user_query)

print("\nRESULT:\n", response)