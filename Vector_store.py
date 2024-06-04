import os
import uuid
import base64
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from unstructured import partition_pdf  # Ensure this is the correct import for your partitioning function

# Load the OpenAI API key from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class VectorStore:
    def __init__(self, pdf_path, output_path):
        """
        Initialize the VectorStore class with paths for the PDF and output directory.
        """
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.raw_pdf_elements = self.partition_pdf()
        self.text_elements = []
        self.table_elements = []
        self.image_elements = []
        self.text_summaries = []
        self.table_summaries = []
        self.image_summaries = []

    def partition_pdf(self):
        """
        Partition the PDF into text, table, and image elements using the partitioning function.
        """
        return partition_pdf(
            filename=self.pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=self.output_path,
        )

    def summarize_text_and_tables(self):
        """
        Summarize text and table elements from the PDF using an LLM chain.
        """
        summary_prompt = """
        Summarize the following {element_type}:
        {element}
        """
        summary_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024),
            prompt=PromptTemplate.from_template(summary_prompt)
        )

        for e in self.raw_pdf_elements:
            if 'CompositeElement' in repr(e):
                self.text_elements.append(e.text)
                summary = summary_chain.run({'element_type': 'text', 'element': e})
                self.text_summaries.append(summary)
            elif 'Table' in repr(e):
                self.table_elements.append(e.text)
                summary = summary_chain.run({'element_type': 'table', 'element': e})
                self.table_summaries.append(summary)

    def encode_image(self, image_path):
        """
        Encode an image in base64 format.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def summarize_image(self, encoded_image):
        """
        Summarize the contents of an image using an LLM.
        """
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the contents of this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        response = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024).invoke(prompt)
        return response.content

    def process_images(self):
        """
        Process and summarize images extracted from the PDF.
        """
        for i in os.listdir(self.output_path):
            if i.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.output_path, i)
                encoded_image = self.encode_image(image_path)
                self.image_elements.append(encoded_image)
                summary = self.summarize_image(encoded_image)
                self.image_summaries.append(summary)

    def create_documents(self):
        """
        Create document objects from text, table, and image summaries.
        """
        documents = []
        retrieve_contents = []

        for e, s in zip(self.text_elements, self.text_summaries):
            doc_id = str(uuid.uuid4())
            doc = Document(
                page_content=s,
                metadata={
                    'id': doc_id,
                    'type': 'text',
                    'original_content': e
                }
            )
            retrieve_contents.append((doc_id, e))
            documents.append(doc)

        for e, s in zip(self.table_elements, self.table_summaries):
            doc_id = str(uuid.uuid4())
            doc = Document(
                page_content=s,
                metadata={
                    'id': doc_id,
                    'type': 'table',
                    'original_content': e
                }
            )
            retrieve_contents.append((doc_id, e))
            documents.append(doc)

        for e, s in zip(self.image_elements, self.image_summaries):
            doc_id = str(uuid.uuid4())
            doc = Document(
                page_content=s,
                metadata={
                    'id': doc_id,
                    'type': 'image',
                    'original_content': e
                }
            )
            retrieve_contents.append((doc_id, s))
            documents.append(doc)

        return documents, retrieve_contents

    def save_vectorstore(self, documents):
        """
        Save the documents into a FAISS vector store.
        """
        vectorstore = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
        vectorstore.save_local("faiss_index")

    def process_pdf(self):
        """
        Complete processing of the PDF, including summarizing text, tables, and images,
        and saving the results to a vector store.
        """
        self.summarize_text_and_tables()
        self.process_images()
        documents, retrieve_contents = self.create_documents()
        self.save_vectorstore(documents)

if __name__ == "__main__":
    # Example usage
    pdf_processor = VectorStore(
        pdf_path="Data/file.pdf",
        output_path="Refined_output/",
    )
    pdf_processor.process_pdf()
