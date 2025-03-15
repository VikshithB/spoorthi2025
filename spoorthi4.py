import os
import warnings
import requests
import json
import re
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import subprocess
import threading
from typing import Tuple
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import torch

# For Google Docs API:
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.docstore.document import Document

# For LangChain vector store & embeddings:
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import Field

# ----------------- Suppress warnings -----------------
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ----------------- Google Docs Integration -----------------
def get_google_doc_text(document_id: str, credentials_file: str) -> str:
    """
    Fetches the entire text content from a Google Doc.
    Replace 'YOUR_API_KEY' with your actual API key if needed.
    """
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    creds = service_account.Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    service = build('docs', 'v1', credentials=creds)
    document = service.documents().get(documentId=document_id).execute()
    content = document.get('body').get('content')
    text = ""
    for element in content:
        if 'paragraph' in element:
            for e in element['paragraph'].get('elements', []):
                text_run = e.get('textRun')
                if text_run and 'content' in text_run:
                    text += text_run['content']
    return text

def build_vectorstore_from_google_doc(embeddings) -> FAISS:
    DOC_ID = "10vkjqz_tP4NOJMhPLjEwRwMlKc2Lui5aupmI66kpci4"    # Replace with your Google Doc ID
    CREDENTIALS_FILE = "credentials.json"  # Path to your service account credentials file
    print("Fetching event information from Google Docs...")
    event_text = get_google_doc_text(DOC_ID, CREDENTIALS_FILE)
    paragraphs = event_text.split("\n\n")
    docs = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]
    print(f"Fetched {len(docs)} document chunks from Google Docs.")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ----------------- STEP 1: Speech-to-Text with Whisper (English only) -----------------
def transcribe_audio(audio_file_path: str, model_size: str = "medium") -> str:
    print("Loading Whisper model (medium, English only)...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    # Load audio and convert it to floating point
    audio = whisper.load_audio(audio_file_path)
    audio = torch.tensor(audio, dtype=torch.float32)  # Convert to float32
    
    result = model.transcribe(audio, task="transcribe")
    return result["text"]

# ----------------- STEP 2: Build Knowledge Base -----------------
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Static content about Spoorthi
static_documents = [
    "Spoorthi is an electrifying annual fest organized by the Department of Electronics and Communication Engineering (ECE)."
    " It brings together students, faculty, and industry professionals to celebrate innovation, creativity, and technical excellence."
    " With technical events, cultural showcases, interactive workshops, and creative exhibitions, Spoorthi is a platform for skill exploration."
    " Date: 16th and 17th April, Venue: ECE Department, JNTUHUCESTH."
]
# Fetch dynamic content from Google Docs
dynamic_document = get_google_doc_text("10vkjqz_tP4NOJMhPLjEwRwMlKc2Lui5aupmI66kpci4", "credentials.json")
documents = static_documents + [dynamic_document]
print("Embedding documents...")
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store and retriever initialized.")

# ----------------- STEP 3: Ollama LLM Wrapper -----------------
class OllamaLLM(LLM):
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:latest"
    temperature: float = 0.2
    max_new_tokens: int = 128
    model_kwargs: dict = Field(default_factory=dict)
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    @property
    def _identifying_params(self):
        return {"base_url": self.base_url, "model": self.model}
    
    def _call(self, prompt: str, stop: list = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            **self.model_kwargs,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
        response.raise_for_status()
        chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode("utf-8"))
                    if "response" in obj:
                        chunks.append(obj["response"])
                    if obj.get("done", False):
                        break
                except Exception:
                    continue
        return "".join(chunks).strip()

print("Initializing Ollama LLM...")
ollama_llm = OllamaLLM()
print("Ollama LLM initialized.")

# ----------------- STEP 4: RetrievalQA Chain -----------------
prompt_template = """You are a friendly, conversational assistant.
Answer the following question concisely using only the context provided.
If the answer is not contained within the context, say "I don't know."
Context: {context}
Question: {question}
Final Answer:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
print("Building RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)
print("RetrievalQA chain built.")

# ----------------- Helper: Normalize Input Keys -----------------
def normalize_qa_input(input_dict: dict) -> dict:
    if "query" in input_dict and "question" not in input_dict:
        input_dict["question"] = input_dict["query"]
    elif "question" in input_dict and "query" not in input_dict:
        input_dict["query"] = input_dict["question"]
    elif "query" not in input_dict and "question" not in input_dict:
        raise ValueError("Input must contain either 'query' or 'question'")
    return input_dict

# ----------------- Helper: Process Answer -----------------
def process_answer(answer_text: str) -> Tuple[str, str]:
    # Only return the final answer; ignore any "thinking" parts.
    match = re.search(r"<think>(.*?)</think>", answer_text, re.DOTALL)
    if match:
        final = answer_text.replace(match.group(0), "").strip()
    else:
        final = answer_text.strip()
    if not final or "I don't know" in final:
        final = "Could you please provide a more specific question related to the information?"
    return "", final

# ----------------- STEP 5: ElevenLabs TTS using mpv -----------------
def text_to_speech_mpv(text: str, api_key: str, voice_id: str, model_id: str, output_filename: str = "response_audio.mp3") -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    subprocess.run(["mpv", output_filename, "--really-quiet"])

# ----------------- UI: Tkinter Chat Application -----------------
class ChatUI:
    def __init__(self, master):
        self.master = master
        master.title("Spoorthi Chatbot")
        self.input_text = ScrolledText(master, height=5, width=60)
        self.input_text.pack(padx=10, pady=5)
        self.output_text = ScrolledText(master, height=10, width=60)
        self.output_text.pack(padx=10, pady=5)
        self.status_label = tk.Label(master, text="Status: Awaiting greeting ('hi'/'hello'/'namaste') to start session.")
        self.status_label.pack(padx=10, pady=5)
    
    def update_input(self, text):
        self.input_text.insert(tk.END, text + "\n")
        self.input_text.see(tk.END)
    
    def update_output(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
    
    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        self.master.update()

# ----------------- Global Variables for Conversation -----------------
session_active = False
empty_count = 0
query_buffer = ""
last_query_time = 0

# ----------------- Continuous Listening and Conversation -----------------
def continuous_listen(ui: ChatUI):
    global session_active, empty_count, query_buffer, last_query_time
    fs = 44100  # 44.1 kHz for clear audio
    snippet_duration = 2  # 2-second snippets for near real-time streaming
    greetings = ["hi", "hello", "namaste"]
    
    while True:
        ui.update_status("Listening...")
        # Record a short snippet in float32 format
        recording = sd.rec(int(snippet_duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        # Save audio to file (WAV file supports float32)
        wav.write("temp.wav", fs, recording)
        
        transcription = transcribe_audio("temp.wav", model_size="medium")
        transcription = transcription.strip()
        ui.update_input(f"User said: {transcription}")
        transcription_lower = transcription.lower()
        
        # Check for global stop command
        if "exit agent" in transcription_lower:
            ui.update_status("Stop command detected. Exiting.")
            break
        
        # If session is not active, check for a greeting to start session
        if not session_active:
            if any(transcription_lower.startswith(g) for g in greetings):
                # Time-based greeting response
                current_hour = time.localtime().tm_hour
                if current_hour < 12:
                    time_greeting = "Good morning"
                elif current_hour < 18:
                    time_greeting = "Good afternoon"
                else:
                    time_greeting = "Good evening"
                agent_greeting = f"{time_greeting}! How can I help you?"
                ui.update_output(f"Agent: {agent_greeting}")
                text_to_speech_mpv(agent_greeting, ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID)
                session_active = True
                query_buffer = ""
                empty_count = 0
                last_query_time = time.time()
            else:
                ui.update_status("Awaiting greeting ('hi'/'hello'/'namaste') to start session...")
        else:
            # Session active: accumulate input
            if transcription_lower:
                query_buffer += " " + transcription_lower
                last_query_time = time.time()
                ui.update_status("Query updated in session.")
            else:
                empty_count += 1
                ui.update_status(f"No input detected. Empty count: {empty_count}")
            
            # If 10 seconds have passed since last input, or 3 successive empty snippets, process the query
            if time.time() - last_query_time > 10 or empty_count >= 3:
                if query_buffer.strip():
                    ui.update_status("Processing your query...")
                    process_current_query(ui, query_buffer)
                else:
                    ui.update_status("No query received. Ending session.")
                session_active = False
                query_buffer = ""
                empty_count = 0
        
        time.sleep(0.5)

def process_current_query(ui: ChatUI, query: str):
    input_data = {"question": query.strip()}
    normalized_input = normalize_qa_input(input_data)
    ui.update_status("Running RetrievalQA chain...")
    raw_answer = qa_chain.invoke(normalized_input)
    _, final_answer = process_answer(raw_answer.get("result", raw_answer))
    ui.update_output(f"Agent: {final_answer}")
    ui.update_status("Converting response to speech...")
    try:
        text_to_speech_mpv(final_answer, ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID)
        ui.update_status("Response spoken.")
    except Exception as e:
        ui.update_output(f"TTS failed: {e}")
        ui.update_status("TTS failed.")

# ----------------- Main Function and UI -----------------
def main():
    global ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID
    # Set your ElevenLabs API key and desired voice/model IDs.
    ELEVENLABS_API_KEY = "sk_d897d0a1131929c79ef50d9a27aa308d6aded0c429ccaef2"
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Example: Rachel
    MODEL_ID = "eleven_multilingual_v2"  # Example model ID
    
    root = tk.Tk()
    ui = ChatUI(root)
    listener_thread = threading.Thread(target=continuous_listen, args=(ui,), daemon=True)
    listener_thread.start()
    root.mainloop()

if __name__ == "__main__":
    main()