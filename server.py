import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

app = FastAPI()

path_vector_store = r"./faiss_index"
llm=ChatOpenAI(model="gpt-4o", max_tokens=1024)

embeddings_model = OpenAIEmbeddings()

faiss_index = FAISS.load_local(path_vector_store, embeddings_model,allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()

prompt_template = """
Du är en Byggassistent,
Expert på Byggledning och Ingenjörskonst.

Besvara frågan så detaljerat som möjligt utifrån den givna kontexten och se till att ge alla detaljer på ett strukturerat sätt. Om svaret inte finns i den givna kontexten, säg bara, 'svaret finns inte i kontexten'. Ge inte ett felaktigt svar. Om svaret är ett ja- eller nej-villkor och innehållet inte finns i den givna kontexten, säg 'Nej', annars säg,
"""
prompt_suffic = """
Kontext:\n {context}?\n
Fråga: \n{question}\n

Svar:
"""
prompt_template_final = prompt_template + prompt_suffic

rag_prompt = ChatPromptTemplate.from_template(prompt_template_final)

entry_point_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
rag_chain = entry_point_chain | rag_prompt | llm | StrOutputParser()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
