import json
import numpy as np
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

'''
Tried the hugging face embeddings but there is some issue in updated langchain-huggingface library ,
even after setting the API token all the models are not getting pulled.
'''
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")



def create_knowledge_base(
        json_file_path : str = "Extracted_Data/extracted_content.json" ,
        # hf_embedding_model : str = "Qwen/Qwen3-Embedding-0.6B",
        ollama_embedding_model : str = "llama3:latest",
        chunk_size : int = 500 , 
        chunk_overlap : int = 50,
        save_directory: str = "./faiss_index"
    ):
    """
    This function loads JSON, split content, generate embeddings, and create FAISS vector store.
    Returns: FAISS vector store.
    """
    ############################# Take the content from extracted data skipping the url #######################
    all_text = ''
    with open(json_file_path , "r" , encoding="utf-8") as f:
        extracted_data = json.load(fp=f)
    for data in extracted_data:
        url = data['url']
        content = data['content']
        all_text += content + "\n"
    


    ############################# Split the concatenated text  #######################
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators=["\n\n" , "\n"]
    )
    splitted_text = text_splitter.split_text(text=all_text)


    ############################# Convert data to vectors #######################
    model_kwargs = {'device': 'cpu'}
    embeddings = OllamaEmbeddings(
        model=ollama_embedding_model
    )

    vector_db = FAISS.from_texts(splitted_text , embeddings)
    vector_db.save_local(save_directory)
    return vector_db



if __name__ == "__main__":

    file_path = "Extracted_Data/extracted_content.json"
    vector_db = create_knowledge_base(json_file_path=file_path)