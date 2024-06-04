import os
import Chat_bot as chatbot
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Define a request model using Pydantic
class QuestionRequest(BaseModel):
    question: str

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI()

llm=ChatOpenAI(model="gpt-4o", max_tokens=1024)

path_vector_store=r"C:\Users\HarshithR\PycharmProjects\erabrajesh\Refined_code\faiss_index"

bot=chatbot.Assistant(chat_model=llm,path=path_vector_store)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chatbot API"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer, relevant_images, video_recommendations = bot.answer(request.question)
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
