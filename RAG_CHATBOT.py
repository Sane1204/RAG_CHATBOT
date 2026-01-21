from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_llm():
    llm=ChatOllama(model="llama3", temperature=0.5,num_predict=256,)
    return llm

def document_loader(file):
    path = file if isinstance(file, str) else file.name
    return PyPDFLoader(path).load()

def text_splitting(data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(data)
    return chunks

def embedding():
    emb=OllamaEmbeddings(model="nomic-embed-text")
    return emb

def vector_database(chunks):
    embedding_model=embedding()
    vectordb= FAISS.from_documents(chunks,embedding_model)
    return vectordb

def retriever(file):
    splits=document_loader(file)
    chunking=text_splitting(splits)
    vectordb=vector_database(chunking)
    retriever=vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_rag_chain(file):
    retriever_1=retriever(file)
    llm=get_llm()

    prompt=ChatPromptTemplate.from_messages([
        ("system","""Answer using ONLY the given context. If not in context, say you don't know.
         and if they ask for predicting the next year questions just give question based on the uploaded dodcument 
         and give the questions like that
         """),
        ("human","Context:\n {context}\n\nQuestion:\n{question}")
    ])
    chain=(
    {
        "context":retriever_1|format_docs,
        "question":RunnablePassthrough()
    }
    |prompt
    |llm
    |StrOutputParser()
)
    return chain

def rag_answer(pdf_file,question):
    if pdf_file is None:
        return "please upload the file"
    if not question or not question.strip():
        return "please type the question."
    
    chain=build_rag_chain(pdf_file)
    return chain.invoke(question)


RAG_CHATBOT=gr.Interface(
    title="RAG CHATBOT",
    fn=rag_answer,
    inputs=[
        gr.File(file_types=[".pdf"],label="Upload Pdf"),
        gr.Textbox(lines=2,label="question",placeholder="type your question here"),
            ],
    outputs=gr.Textbox(lines=10,label="answer"),
    allow_flagging="never",
)
RAG_CHATBOT.launch(show_api=False)





