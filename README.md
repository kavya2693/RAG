# Chatbot Application Readme



## Please download model files from hugging face and in `models` folder before running it. And rename model path in `main.py`.
Link to Zypher 7B model: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/blob/main/zephyr-7b-beta.Q4_K_M.gguf
Link to 7B Llama 2 model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf 


if you want to use openai's LLM models. 

## Plese provide your openai API key in `.env` (change copy.env to .env first) to use openai models instead.`

## Please add your files [pdf, docx] in 'files' folder


## Running the Application

Follow these steps to set up and run the entire application:

1. **Create a Virtual Environment:**
   
   python -m venv venv
 

2. **Activate the Virtual Environment:**
   - On Windows:
    
     .\venv\Scripts\activate
     
   - On Linux/Mac:
    
     source venv/bin/activate
    
- IF I WANT TO RUN AGAIN: THEN USE LINE 33 - 
## only for local llm models

3. `Recommended for macbook`:  install ollama (https://ollama.com/download) and make changes in main.py if ollama is not selected already. Find it between line 60 and 70 in `main.py` 
run ' ollama run llama2' in terminal


or 


`Install CMAKE` https://cmake.org/download/ This is required to run local llm models
While installing, check 'add to system variables' option. 

follow these instructions to tranfer llm model to `GPU or CPU`:
https://python.langchain.com/docs/integrations/llms/llamacpp
run the command in terminal without ! sign.

then run 

pip install llama-cpp-python



4. **Install Required Modules:**
  
   pip install -r requirements.txt

5. **Ingest Data:**

## Rename `Streamlit.streamlit` to `.streamlit` for app Theme. 
6. **Run Streamlit App:**
   - Start the Streamlit app with `streamlit run app.py` to use a graphical interface for the chatbot.

Note: Ensure that each step is executed in order for the proper functioning of the application.


## files only this 'python data_ingest.py' 'streamlit run app.py'

### 1. `data_ingest.py`
- **Purpose:** This file will imbed the data and create vector database from your files in `files` folder. Add unversity related documetns in files filder.
- **How to Run:** run it with `python data_ingest.py`  or `python3 data_ingest.py` in the terminal.


### 2. `main.py`
- **Purpose:** This file contains the main function of the chatbot. This can be used in frontend application for making call to chatbot.
- **How to Run:** Run it with `python data_ingest.py`  or `python3 data_ingest.py` in the terminal.


### 2. `restapi_app.py`
- **Purpose:** This file is the RestAPI app. This can be used in frontend application in javascript for making call to chatbot.
- **How to Run:** Run it with `python restapi_app.py`  or `python3 restapi_app.py` in the terminal. It will run the app on `http://127.0.0.1:5000/chatbot` for api call


### 3. `app.py`
- **Purpose:** This file contains a Streamlit application for a user-friendly interface to interact with the chatbot.
- **How to Run:** Start the Streamlit app by running `streamlit run app.py` in the terminal.





