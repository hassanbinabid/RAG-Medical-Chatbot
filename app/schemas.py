from pydantic import BaseModel

# ===============================
# Request Model
# ===============================

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of diabetes?"
            }
        }

# ===============================
# Response Model
# ===============================

class AnswerResponse(BaseModel):
    question: str
    answer: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of diabetes?",
                "answer": "The symptoms of diabetes include frequent urination, excessive thirst..."
            }
        }