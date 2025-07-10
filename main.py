from web_crawler import crawl_website, save_to_json
from create_knowledge_base import create_knowledge_base
import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize Streamlit session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app layout
st.title("Website RAG App")

# URL input and crawling
input_url = st.text_input("Enter URL:", placeholder="https://dev.algorand.co/getting-started/introduction/")
if st.button(label="Start"):
    if input_url:
        with st.spinner("Crawling website..."):
            extracted_data = crawl_website(start_url=input_url)
            save_to_json(data=extracted_data)
            st.success("Website crawling successful!")
        
        with st.spinner("Building knowledge base..."):
            st.session_state.vector_db = create_knowledge_base(json_file_path="Extracted_Data/extracted_content.json")
            st.success("Knowledge database ready!")
    else:
        st.error("Please provide a URL...")

st.subheader("Ask a Question")
query = st.text_input("Enter your question:", key="query_input")
if st.button(label="Submit Query"):
    if not query:
        st.error("Please enter a question.")
    elif st.session_state.vector_db is None:
        st.error("Please crawl a website and build the knowledge base first.")
    else:
        try:
            with st.spinner("Generating response..."):
                # Initialize embeddings and Ollama model
                embeddings = OllamaEmbeddings(model="llama3:latest")
                llm = ChatOllama(model="llama3:latest")
                
                # Define prompt template
                prompt_template = ChatPromptTemplate.from_template(
                    """
                    You are a helpful assistant. Using the provided context, answer the user's question concisely and accurately.
                    If the context is insufficient, say so and provide a general answer.
                    
                    **Context**: {context}
                    
                    **Question**: {query}
                    
                    **Answer**:
                    """
                )
                
                # Retrieve top 3 relevant chunks
                docs = st.session_state.vector_db.similarity_search(query, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Format prompt and generate response
                prompt = prompt_template.format(context=context, query=query)
                response = llm.invoke(prompt)
                
                # Store in chat history
                st.session_state.chat_history.append({"query": query, "answer": response.content})
                
                # Display response
                st.success("Response generated!")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        with st.container():
            st.markdown(f"**You**: {chat['query']}")
            st.markdown(f"**Assistant**: {chat['answer']}")
            st.markdown("---")