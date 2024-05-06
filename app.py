import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
         if i % 2 == 0:
            st.write(f"User: {message.content}")
         else:
            st.write(f"Bot: {message.content}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Progress bar for getting PDF text
                progress_text = st.progress(0)
                progress_text.text('Extracting PDF text...')
                raw_text = get_pdf_text(pdf_docs)
                progress_text.progress(50)

                # Progress bar for getting text chunks
                progress_chunks = st.progress(0)
                progress_chunks.text('Splitting text into chunks...')
                text_chunks = get_text_chunks(raw_text)
                progress_chunks.progress(75)

                # Progress bar for creating vector store
                progress_vectorstore = st.progress(0)
                progress_vectorstore.text('Creating vector store...')
                vectorstore = get_vectorstore(text_chunks)
                progress_vectorstore.progress(90)

                # Progress bar for creating conversation chain
                progress_conversation = st.progress(0)
                progress_conversation.text('Creating conversation chain...')
                st.session_state.conversation = get_conversation_chain(vectorstore)
                progress_conversation.progress(100)

if __name__ == '__main__':
    main()
