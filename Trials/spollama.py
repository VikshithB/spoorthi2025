import os
import requests
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM

# ---------------------------
# Gemini API configuration
# ---------------------------
# Replace these with your actual Gemini API settings.
GEMINI_PROJECT_ID = "spoorthi25"           # Your project ID (example)
GEMINI_LOCATION = "us-central1"            # Your region (example)
GEMINI_MODEL_ID = "gemini-2.0-flash-lite"    # Use the supported model ID (e.g., Gemini 2.0 Flash-Lite)
GEMINI_API_KEY = "AIzaSyBH7LKM26_5EIPnPPU0LG2Z4tzWRIVH1Zw"       # Your Gemini API key

# Construct the Gemini API endpoint URL.
GEMINI_API_ENDPOINT = (
    f"https://vertexai.googleapis.com/v1/projects/{GEMINI_PROJECT_ID}/locations/{GEMINI_LOCATION}/publishers/google/models/{GEMINI_MODEL_ID}:predict"
)

def gemini_generate(prompt, max_tokens=256, temperature=0.7):
    """
    Calls the Gemini API endpoint with the given prompt and returns the generated output.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    data = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(GEMINI_API_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Assuming the response structure is:
    # {"predictions": [{"output": "generated text"}]}
    return result["predictions"][0]["output"]

# ---------------------------
# Custom LLM Wrapper for Gemini API
# ---------------------------
class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini_api"
    
    def _call(self, prompt: str, stop=None) -> str:
        return gemini_generate(prompt)
    
    @property
    def _identifying_params(self):
        return {"model_id": GEMINI_MODEL_ID, "project_id": GEMINI_PROJECT_ID}

# ---------------------------
# Google Docs integration
# ---------------------------
def get_google_doc_text(document_id, credentials_file):
    """
    Fetches the entire text content from a Google Doc.
    """
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    creds = service_account.Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    service = build('docs', 'v1', credentials=creds)
    document = service.documents().get(documentId=document_id).execute()
    content = document.get('body').get('content')
    text = ""
    for element in content:
        if 'paragraph' in element:
            elements = element['paragraph'].get('elements', [])
            for e in elements:
                text_run = e.get('textRun')
                if text_run and 'content' in text_run:
                    text += text_run['content']
    return text

def build_vectorstore_from_google_doc():
    """
    Fetches event info from the Google Doc and builds a FAISS vector store.
    """
    DOC_ID = "10vkjqz_tP4NOJMhPLjEwRwMlKc2Lui5aupmI66kpci4"    # Replace with your actual Google Doc ID
    CREDENTIALS_FILE = "credentials.json"  # Path to your service account credentials JSON file

    print("Fetching event information from Google Docs...")
    event_text = get_google_doc_text(DOC_ID, CREDENTIALS_FILE)
    # Split the text into paragraphs (using double newlines as separator)
    paragraphs = event_text.split("\n\n")
    docs = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]
    print(f"Fetched {len(docs)} document chunks from Google Docs.")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ---------------------------
# Set up embeddings and vector store
# ---------------------------
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model loaded.")

# ---------------------------
# Retrieval and QA Chain using Gemini API
# ---------------------------
def retrieve_context(query, vectorstore, k=3):
    """
    Retrieves the top k relevant document chunks based on the query.
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def answer_query(query, vectorstore):
    """
    Retrieves context for the query and uses the Gemini API (via our custom GeminiLLM)
    to generate an answer.
    """
    retrieved_docs = retrieve_context(query, vectorstore)
    
    # Instantiate our custom Gemini LLM wrapper.
    llm = GeminiLLM()
    
    # Create a QA chain using the "stuff" method.
    chain = load_qa_chain(llm, chain_type="stuff")
    print("Generating answer...")
    answer = chain.run(input_documents=retrieved_docs, question=query)
    return answer

# ---------------------------
# Main interactive loop
# ---------------------------
if __name__ == "__main__":
    # Build (or refresh) the vector store from the latest Google Doc content.
    vectorstore = build_vectorstore_from_google_doc()
    
    print("\nEvent information loaded from Google Docs.")
    print("You can now ask questions about the event (type 'exit' to quit).")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        answer = answer_query(query, vectorstore)
        print("\nAnswer:", answer)
