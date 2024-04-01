from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
import chromadb
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


folder_path = "./files"

class make_vector_store():
    
 

    def create_vectordb(folder_path):
        
        loader = DirectoryLoader(folder_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        docs = splitter.split_documents(documents)
        persist_directory = 'chroma_db'
        embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large",model_kwargs = {'device': 'cpu'})
        vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        
        vectordb.persist()
        
        return vectordb

if __name__ == "__main__":
    make_vector_store.create_vectordb(folder_path)

