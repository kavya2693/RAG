import os

import streamlit as st
from PyPDF2 import PdfReader
import langchain
from textwrap import dedent
import pandas as pd
from langchain_openai.llms import OpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import shutil
import os

from main import RAGbot

from datetime import date

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores.chroma import Chroma

import chromadb

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings




date_today = date.today()



st.set_page_config(page_title='Personal Chatbot', page_icon='books')



# Add CSS targeting both the image and its container



st.markdown(
    """
    <style>
        [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: 10%;
            margin-right:10%;
            width: 100%;
    }
    img {
        border-radius: 150%;
        align: center;
    }
    </style>
    """, unsafe_allow_html=True
)


st.markdown("<h6 style='text-align: center; color: black;'> Document Chatbot </h6>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 0.85rem;
        color: #000000

    }
    </style>
    """, unsafe_allow_html=True
)
# Reducing whitespace on the top of the page
st.markdown("""
<style>

.block-container
{
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def process_pdf_docx(uploaded_file):
    with st.spinner(text="Embedding Your Files"):

        # Read text from the uploaded PDF file
        data = []
        for file in uploaded_file:
            split_tup = os.path.splitext(file.name)
            file_extension = split_tup[1]
        
            if file_extension == ".pdf":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file1:
                    tmp_file1.write(file.getvalue())
                    tmp_file_path1 = tmp_file1.name
                    loader = PyPDFLoader(file_path=tmp_file_path1)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    data += text_splitter.split_documents(documents)

            if file_extension == ".docx":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                    loader = UnstructuredWordDocumentLoader(file_path=tmp_file_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                    data += text_splitter.split_documents(documents)
                

       
        
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",model_kwargs = {'device': 'cpu'})
        vectordb = Chroma.from_documents(documents=data, embedding=embeddings)
        
       
        return vectordb


with st.sidebar:
    uploaded_file =  st.file_uploader("Upload your files",
    help="Multiple Files are Supported",
    type=['pdf', 'docx'], accept_multiple_files= True)




if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"]= []


def main():
    
    try:
        if uploaded_file:
            db = process_pdf_docx(uploaded_file)
        else: 
            db= None
            
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "User", ai_prefix= "Assistant")
        for msg in st.session_state.messages:
            
            if msg["role"]== 'Assistant':
                st.chat_message(msg["role"], avatar="logo_bot.png").write(msg["content"])
            else: 
                st.chat_message(msg["role"], avatar = "user.png").write(msg["content"])
                
                
        #prompt = prompt
        if prompt := st.chat_input(placeholder="Type your question!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    user_prompt = st.session_state.messages[i]
                    ai_res = st.session_state.messages[i + 1]
                    
                
                    user_content = user_prompt["content"]
                    
                    
                    ai_content = ai_res["content"]
                    
                    # Concatenate role and content for context and output
                    user = f"{user_content}"
                    ai = f"{ai_content}"
                    
                    memory.save_context({"question": user}, {"output": ai})
        
            
            st.chat_message("user", avatar = "user.png").write(prompt)
             
          
            with st.chat_message("Assistant", avatar= "logo_bot.png"):
                with st.spinner('Assistant...'):
                    chatbot= RAGbot
                    response = chatbot.run(prompt, db, memory)
                    st.session_state.messages.append({"role": "Assistant", "content": response})
                    st.write(response) 
            #memory.save_context({"question": prompt}, {"output": response})
                           
                                          
    except Exception as e:
        with st.chat_message("Assistant", avatar= "logo_bot.png"):
            
            st.write("Sorry, the question is irrelevant or the bot crashed")
 



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    main()





