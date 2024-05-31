
import os
from dotenv import load_dotenv
load_dotenv()

# load API Keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['ASTRA_DB_API_ENDPOINT'] = os.getenv('ASTRA_DB_API_ENDPOINT')
os.environ['ASTRA_DB_APPLICATION_TOKEN'] = os.getenv('ASTRA_DB_APPLICATION_TOKEN')

import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader

def vector_embeddings():    
    st.session_state.embeddings = HuggingFaceBgeEmbeddings()
    collection_name = os.getenv("ASTRA_DB_COLLECTION")
    st.session_state.vectorStore = AstraDBVectorStore(
        collection_name = collection_name,
        embedding = st.session_state.embeddings,
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    )
    vectorIndex = st.session_state.vectorStore.astra_db.collection(collection_name).find()
    if len(vectorIndex) == 0:
        st.session_state.loader = PyPDFDirectoryLoader("./Data") # data ingestion
        st.session_state.documents = st.session_state.loader.load() # loading documents
        st.session_state.vectorStore.add_documents(st.session_state.documents) # create and add vectors to Astra DB


vector_embeddings()


import time

st.title("Product Information Assistant")


llm = ChatGroq(groq_api_key=groq_api_key, model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}

    """
    )
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.vectorStore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

question = st.text_input("Enter your questions on the Trapeze Products and Solutions")

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




