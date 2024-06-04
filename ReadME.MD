# Multimodel Q&A for Construction

## Overview

Multimodel Q&A for Construction is an application that enhances the question and answer process by providing suggestions from multiple sources. This includes text, images, and video suggestions from YouTube, all of which are related to the given query. The application leverages the FAISS (Facebook AI Similarity Search) vector database to manage and retrieve relevant information from both text and image data.

## Features

- **Text Suggestion**: Provides text-based answers and suggestions related to the query.
- **Image Suggestion**: Retrieves related images from a vector database.
- **Video Suggestion**: Searches YouTube for relevant video content and provides suggestions.

## Prerequisites

To run this prototype, you will need:

- An OpenAI API key.
- A YouTube Data API key.
- FAISS index file path configured in the `app.py` file.

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and Configure the `.env` File**:
    Create a `.env` file in the root directory of the project and add your OpenAI API key and YouTube API key.
    ```
    OPENAI_API_KEY=your_openai_api_key
    DEVELOPER_KEY=your_youtube_api_key
    ```

3. **Check FAISS Index Path**:
    Ensure the path of the FAISS index is correctly defined in the `app.py` file.
    ```python
    # Example configuration in app.py
    FAISS_INDEX_PATH = 'path_to_your_faiss_index'
    ```

4. **Install Dependencies**:
    Install the required Python packages using pip.
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the API**:
    Use Uvicorn to start the API server.
    ```sh
    uvicorn app:app --reload
    ```

6. **Test the API**:
    Use the `client.py` file to test the API.
    ```sh
    python client.py
    ```

## Usage

Once the server is running, you can send queries to the API and receive text, image, and video suggestions related to your query.

### Example Query

Send a request to the endpoint with your query in text format. The API will return responses that include:

- Text suggestions relevant to the query from Vector index.
- Images related to the query retrieved from the Vector index.
- YouTube video links related to the query.

---

# Note
  * You can also create your own vector index using `vector_store.py`.



