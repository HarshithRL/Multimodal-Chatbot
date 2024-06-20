import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

class Assistant:
    def __init__(self, chat_model,path):
        """
        Initialize the Assistant class with a chat model and the path to the vector index.
        """
        self.chat_model = chat_model  # Open AI gpt-4o is suggested
        self.path=path # Path of the vector Index
        self.vectore = self.get_vector_text()


    def get_vector_text(self):
        """
        Load the FAISS vector store from the specified local path.

        """
        vector_store = FAISS.load_local(
            self.path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True) # Allow dangerous deserialization because we are using a local version of the FAISS index
        return vector_store.as_retriever()

    def get_conve_chain(self):
        """
        Create an LLMChain for generating answers based on the provided context and question.
        """
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
        prompt = PromptTemplate(template=prompt_template_final, input_variables=["context", "question"],
                                callbacks=[StrOutputParser])
        llm = self.chat_model
        output_parser=StrOutputParser()
        # chain = load_qa_chain(self.chat_model, chain_type="stuff", prompt=prompt)
        # chain = LLMChain(llm=self.chat_model, prompt=prompt)
        chain=prompt | llm | output_parser
        return chain


    def get_video_recommendations(self,query, max_results=5):
        """
        Fetch video recommendations from YouTube based on the query.
        """
        llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)
        optimizer_query = llm.invoke(f'Give me a relevant search title for the given question in sewden in english to search in youtube {query}')
        youtube = build('youtube', 'v3', developerKey=os.getenv('DEVELOPER_KEY'))
        search_response = youtube.search().list(
            q=optimizer_query.content,
            part='snippet',
            maxResults=max_results,
            type='video'
        ).execute()

        videos = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            videos.append(f"https://www.youtube.com/watch?v={video_id} - {title}")

        return videos

    def answer(self,question):
        """
        Answer the given question by retrieving relevant documents from the vector store,
        generating a detailed response, and fetching related video recommendations .
        """
        # Retrieve relevant documents from the vector store
        relevant_docs = self.vectore.invoke(question)
        context = ""
        relevant_images = []
        chain = self.get_conve_chain()

        # Compile context from the retrieved documents
        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
            elif d.metadata['type'] == 'image':
                context += d.page_content
                relevant_images.append(d.metadata['original_content'])

        # Generate the answer using the LLMChain
        result = chain.invoke({'context': context, 'question': question})
        # Fetch video recommendations from YouTube
        video_recommendations = self.get_video_recommendations(f"find the video related to {question}")

        return result, relevant_images,video_recommendations







