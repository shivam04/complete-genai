import json
import os
import sys
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock

## Data Ingestion

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLm Models
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Anthropic Model
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={
            "max_tokens": 512,
        },
    )

    return llm

def get_llama3_llm():
    llama3_llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={
            "max_tokens": 512,
        },
        client=bedrock
    )

    return llama3_llm

prompt_template = """
Human:
You are a research assistant.

Use the retrieved context to answer the question.
If the context is partial, synthesize a coherent explanation
based on the available information, and clearly state any limitations.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore,query):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )
    chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    answer = chain.invoke(query)
    return answer

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    if st.button("Claude Output"):
        if not user_question:
            st.warning("Please enter a question")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")
    if st.button("Llama Output"):
        if not user_question:
            st.warning("Please enter a question")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()