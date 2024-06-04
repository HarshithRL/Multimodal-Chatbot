import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

class Assistant:
    def __init__(self, chat_model,path):
        self.chat_model = chat_model
        self.path=path
        self.vectore = self.get_vector_text()


    def get_vector_text(self):
        vector_store = FAISS.load_local(self.path, embeddings=OpenAIEmbeddings(),
                              allow_dangerous_deserialization=True)
        return vector_store

    def get_conve_chain(self):
        prompt_template = """
        You are a Construction Assistant,
        Expert in Construction Management and Engineering.

        Answer the Question as detailed as possible from the provided context, making sure to provide all the details in a structured way. If the answer is not in the provided context, just say, 'answer is not available in the context'. Do not provide a wrong answer. If the answer is a yes or no condition and the content is not in the provided context, say 'No', or else say,
        """
        prompt_suffic = """
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt_template_final = prompt_template + prompt_suffic
        prompt = PromptTemplate(template=prompt_template_final, input_variables=["context", "question"],
                                callbacks=[StrOutputParser])

        # chain = load_qa_chain(self.chat_model, chain_type="stuff", prompt=prompt)
        chain = LLMChain(llm=self.chat_model, prompt=prompt)
        return chain


    def get_video_recommendations(self,query, max_results=5):
        youtube = build('youtube', 'v3', developerKey=os.getenv('DEVELOPER_KEY'))
        search_response = youtube.search().list(
            q=query,
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
        relevant_docs = self.vectore.similarity_search(question)
        context = ""
        relevant_images = []
        chain = self.get_conve_chain()
        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
            elif d.metadata['type'] == 'image':
                context += d.page_content
                relevant_images.append(d.metadata['original_content'])
        result = chain({'context': context, 'question': question}, return_only_outputs=True)
        video_recommendations = self.get_video_recommendations(f"find the video in english related to {question}")
        return result, relevant_images,video_recommendations

# llm=ChatOpenAI(model="gpt-4o", max_tokens=1024)
# path_vector_store=r"C:\Users\HarshithR\PycharmProjects\erabrajesh\Refined_code\faiss_index"
# result, relevant_images,video_recommendations=Assistant(chat_model=llm,path=path_vector_store).answer('Types of design foundation')





