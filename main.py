import os
import langchain
from textwrap import dedent
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models import ChatGooglePalm
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate
)
import openai

from vectordbsearch_tools import VectorSearchTools_chroma

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") #OpenAI API Key

llm_openai = ChatOpenAI(temperature =0.1, max_tokens= 1024,  model="gpt-3.5-turbo-1106") 

#for linux or mac with litellm and ollama
lite_llm = ChatOpenAI(
    openai_api_base="http://0.0.0.0:8000",
    model = "llama",
    api_key= "h",
    temperature=0.1,
    
)

def load_local_model(model_path):
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 10  # Increase based on your GPU capacity. 0 means no gpu usage.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM.
 
    llm_llama = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        temperature=0.1,
        n_batch=n_batch,
        n_ctx=8000,
        max_tokens= 1024,
        top_k =5,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm_llama
       
ollama_llm = Ollama(model="mistral", num_ctx=8096, temperature=0.1, top_k=5, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
#install Ollama application and run with 'ollama run 'model name''. for example: ollama run llama2


#model_path="./models/zephyr-7b-alpha.Q4_K_M.gguf"  #local model file. add model file (.gguf) in model models folder and replace the name/path

#local_llm = load_local_model(model_path)  


llm=llm_openai #you can use whichever model you want (e.g ollama_llm, openai_llm or local_llm)


class RAGbot:
      
    
    def run(prompt, db, memory):
       
        #try:  
               
          
          # Get user input -> Generate the answer
          greetings = ['Hey', 'Hello', 'hi', 'hu' 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                      'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you',
                      'its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                      'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you",
                      'how are things', "how're things", 'how is it going', "how's it going", "how's it goin'", "how's it goin",
                      'how is life been treating you', "how's life been treating you", 'how have you been', "how've you been",
                      'what is up', "what's up", 'what is cracking', "what's cracking", 'what is good', "what's good",
                      'what is happening', "what's happening", 'what is new', "what's new", 'what is neww', "g’day", 'howdy' 'ji']
          compliment = ['thank you', 'thanks', 'thank' 'thanks a lot', 'thanks a bunch', 'great', 'ok', 'ok thanks', 'okay', 'great', 'good' 'awesome', 'nice']
                      
          prompt_template =dedent(r"""
              Use the following pieces of context to answer the question at the end. Answer in the same language of the question.
            
              Do not Answer from your training data. 
              If the Answer is not found in the context. Do not make up any answer. Do not make up hypothetical answers. Answer the correct information only.
              Answer correcty and to the point. Do not add unwanted information.
              Always answer in the same language of the question.
              Ensure that your answers are directly related to the user's question and chat history.
             
              This is the context:
              ---------
              {context}
              
            
              
              {uploaded_files}
              ---------

              chat history: 
              ---------
              {chat_history}
              ---------

              Question: 
              {question}

              Helpful Answer: 
              """)
              
              

          PROMPT = PromptTemplate(
                  template=prompt_template, input_variables=[ "context", "uploaded_files", "question", "chat_history"]
              )

            
          chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                  
        
        
          if prompt.lower() in greetings:
            response = 'Hi, how are you? I am here to help you get information from your documents. How can I assist you?'
            
              
            return response
              
          elif prompt.lower() in compliment:
            response = 'My pleasure! If you have any more questions, feel free to ask.'
            
              
            return response
            #memory.save_context({"question": prompt}, {"output": response})
              
          else:
            
            docs =VectorSearchTools_chroma.dbsearch(prompt)
            if db is not None:
                uploaded_files= db.similarity_search(prompt, k=4)
                plain_texts = [f"\n{doc.page_content} {doc.metadata}" for doc in uploaded_files]
                merged_texts = '\n'.join(plain_texts)
            
                response = chain.run(input_documents=docs, uploaded_files= merged_texts, question=prompt)
            else:
                uploaded_files= []
            
                response = chain.run(input_documents=docs, uploaded_files= uploaded_files, question=prompt)
            
          
              
            return response
        
        #except Exception as e:
            
         #   "Sorry, the question is irrelevant or the bot crashed"
    


if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        chatbot= RAGbot
        result = chatbot.run(query)
        print("Bot:", result)