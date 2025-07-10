# Web RAG Chatbot


# Overview
- This project is a Python-based Retrieval-Augmented Generation (RAG) chatbot that crawls a website, builds a local knowledge base, and answers user queries using a local language model. It features two interfaces: a FastAPI server for programmatic access and a Streamlit app for an interactive web-based chat experience. The chatbot extracts content from a given website, processes it into a searchable knowledge base using vector embeddings, and generates responses with Ollama's local language model (llama3).




1. Setup Instructions

Create a Virtual Environment:

- python -m venv .venv
###### OR
- uv venv


2. Activate the Virtual Environment:

 - Windows:.venv\Scripts\activate





3. Install Dependencies:
- uv sync
###### OR
- uv pip install -r requirements.txt




4. Run the Application:

- FastAPI Server : uv run uvicorn chatbot:app --port {PORT_NUMBER} , Access the API at http://localhost:{PORT_NUMBER}/docs.
##### OR
- Streamlit App : streamlit run app.py



# Project Structure

- web_crawler.py: Crawls a website using requests and BeautifulSoup, saving content to Extracted_Data/extracted_content.json.
- knowledge_base.py: Loads JSON, splits content, generates embeddings with OllamaEmbeddings, and builds a FAISS vector store.
- chatbot.py: FastAPI app with a /ask endpoint for querying the chatbot using Ollama.
- app.py: Streamlit app for crawling websites, building the knowledge Melody and answering queries with chat history.
- Extracted_Data/: Directory for storing extracted_content.json.
- faiss_index/: Directory for FAISS vector store.

# Usage

- Use the Streamlit app and enter a URL to crawl and build the knowledge base.
Ask questions via the chat interface or FastAPI /ask endpoint.
View chat history in the Streamlit app.

