from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag import get_answer
from app.schemas import QuestionRequest, AnswerResponse

# ===============================
# FastAPI App
# ===============================

app = FastAPI(
    title="ClinicaBot API 🏥",
    description="Medical Chatbot API powered by RAG (Retrieval-Augmented Generation) based on Top Medicine Encyclopedias.",
    version="1.0.0"
)

# ===============================
# CORS Middleware
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Routes
# ===============================

@app.get("/")
def health_check():
    return {"status": "online", "message": "Welcome to ClinicaBot API 🏥"}


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return AnswerResponse(
            question=request.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))