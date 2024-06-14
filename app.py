import os
import Chat_bot as chatbot
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the OpenAI API key from environment variables
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Define a request model using Pydantic
class QuestionRequest(BaseModel):
    question: str


# Define a FastAPI object

app=FastAPI()

# Initialize the OpenAI GPT-4o model

llm=ChatOpenAI(model="gpt-4o", max_tokens=1024)

# Path of the Vector Index

path_vector_store=r"\faiss_index"


# Instantiate the Assistant class by chaining both Vector Index & Chat Model

bot=chatbot.Assistant(chat_model=llm,path=path_vector_store)


# Welcome route

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chatbot API"}

# Route to handle questions

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer, relevant_images,video_recommendations= bot.answer(request.question)

        return {
            "answer": answer,
            "relevant_images": relevant_images,
            "video_recommendations": video_recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#uvicorn app:app --reload
