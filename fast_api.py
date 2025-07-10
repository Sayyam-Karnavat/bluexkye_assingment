from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings , ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Request model for /ask endpoint
class QueryRequest(BaseModel):
    query: str


llm = ChatOllama(model="llama3:latest")
embeddings = OllamaEmbeddings(model="llama3:latest")

# Load FAISS vector store
vector_db = FAISS.load_local(
    folder_path="./faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Using the provided context, answer the user's question concisely and accurately.
    If the context is insufficient, say so and provide a general answer.
    
    Context: {context}
    
    Question: {query}
    
    Answer:
    """
)

@app.get("/")
def homepage():
    return "Server is running !!!"

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Handle user query, retrieve relevant context, and generate response using Ollama.
    """
    try:
        # Retrieve top 3 relevant chunks
        docs = vector_db.similarity_search(request.query, k=3)        
        context = "\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        prompt = prompt_template.format(context=context, query=request.query)
        
        # Generate response with Ollama
        response = llm.invoke(prompt)
        
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")