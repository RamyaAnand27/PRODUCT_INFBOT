
import os
from dotenv import load_dotenv
load_dotenv()

# load API Keys
groq_api_key = os.getenv('GROQ_API_KEY')

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

def vector_embeddings():
    if "vectors" not in st.session_state:
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        st.session_state.loader = PyPDFDirectoryLoader("./Data") # data ingestion
        st.session_state.documents = st.session_state.loader.load() # loading documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # chunk creation
        st.session_state.split_documents = st.session_state.text_splitter.split_documents(st.session_state.documents) # document splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.split_documents, st.session_state.embeddings) # create vector embeddings
      

vector_embeddings()


st.title("Product Information Assisstant")


llm = ChatGroq(groq_api_key=groq_api_key, model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """
    )
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

question = st.text_input("Enter your questions on the Trapeze group Products and solutions...")

if question:
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input':question})
    print("Response time:", time.process_time()-start_time)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        # find the context
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------------")





