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
from typing import Tuple
import torch

# Suppress FP16 warnings from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def record_audio(duration=5, sample_rate=16000):
    print("Recording snippet...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait for recording to complete
    return np.squeeze(audio_data)  # Convert to 1D array if needed

# -------------------- Google Docs Integration --------------------
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.docstore.document import Document

def get_google_doc_text(document_id: str, credentials_file: str) -> str:
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

def build_vectorstore_from_google_doc(embeddings) -> 'FAISS':
    """
    Fetches event info from the Google Doc and builds a FAISS vector store.
    """
    DOC_ID = "10vkjqz_tP4NOJMhPLjEwRwMlKc2Lui5aupmI66kpci4"    # Replace with your actual Google Doc ID
    CREDENTIALS_FILE = "credentials.json"  # Path to your service account credentials JSON file

    print("Fetching event information from Google Docs...")
    event_text = get_google_doc_text(DOC_ID, CREDENTIALS_FILE)
    paragraphs = event_text.split("\n\n")
    docs = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]
    print(f"Fetched {len(docs)} document chunks from Google Docs.")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# -------------------- STEP 1: Speech-to-Text with Whisper --------------------
def transcribe_audio(audio_file_path: str, model_size: str = "medium") -> str:
    print("Loading Whisper model (medium)...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    # Load audio and convert it to floating point
    audio = whisper.load_audio(audio_file_path)
    audio = torch.tensor(audio, dtype=torch.float32)  # Convert to float32
    
    result = model.transcribe(audio, task="transcribe")
    return result["text"]

# -------------------- STEP 2: Knowledge Base & Vector Store Setup --------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = build_vectorstore_from_google_doc(embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store and retriever initialized.")

# -------------------- STEP 3: Custom Ollama LLM Wrapper --------------------
from langchain.llms.base import LLM
from pydantic import Field

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

# -------------------- STEP 4: Prompt Template & RetrievalQA Chain --------------------
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """You are an expert assistant.
Answer the following question concisely using only the context provided.
If the answer is not contained within the context, say "I don't know.
Keep the answer to the point and avoid unnecessary details.
Give the answer in 2-3 sentences maximum. until it is necessary to give the answer in more detail."
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

# -------------------- Helper: Normalize Input Keys --------------------
def normalize_qa_input(input_dict: dict) -> dict:
    """
    Ensure that both 'query' and 'question' keys are present and have the same value.
    """
    if "query" in input_dict and "question" not in input_dict:
        input_dict["question"] = input_dict["query"]
    elif "question" in input_dict and "query" not in input_dict:
        input_dict["query"] = input_dict["question"]
    elif "query" not in input_dict and "question" not in input_dict:
        raise ValueError("Input must contain either 'query' or 'question'")
    return input_dict

# -------------------- Helper: Process Answer --------------------
def process_answer(answer_text: str) -> Tuple[str, str]:
    """
    Extract the 'thinking' part from within <think>...</think> tags and separate the final answer.
    Only the final answer is used for TTS.
    """
    thinking = ""
    final = answer_text
    match = re.search(r"<think>(.*?)</think>", answer_text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        final = answer_text.replace(match.group(0), "").strip()
    if not final or "I don't know" in final:
        final = "Could you please provide a more specific question related to the information?"
    return thinking, final                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

# -------------------- STEP 5: ElevenLabs TTS using mpv --------------------
def text_to_speech_mpv(text: str, api_key: str, voice_id: str, model_id: str, output_filename: str = "response_audio.mp3") -> None:
    """
    Get TTS audio from ElevenLabs, save to a file, and play it using mpv.
    """
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
    
    # Play the audio file using mpv
    subprocess.run(["mpv", output_filename, "--really-quiet"])

# -------------------- STEP 6: Continuous Conversation with Session Management --------------------
def run_chatbot():
    # Set your ElevenLabs API key and desired voice/model IDs.
    ELEVENLABS_API_KEY = "sk_d897d0a1131929c79ef50d9a27aa308d6aded0c429ccaef2"
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Example voice (Rachel)
    MODEL_ID = "eleven_multilingual_v2"  # Example model ID

    # Set sampling rate to 44100 for clear audio
    fs = 44100
    snippet_duration = 5  # Record 5-second snippets

    session_active = False
    empty_count = 0  # Count successive empty inputs
    print("Awaiting wake word 'hi agent' to start session...")
    
    while True:
        # Record a snippet
        print("\nRecording snippet...")
        recording = sd.rec(int(snippet_duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write("temp.wav", fs, recording)
        
        transcription = transcribe_audio("temp.wav", model_size="medium")
        print("Transcription:", transcription)
        
        transcription_lower = transcription.lower().strip()

        # Always check for stop word
        if "exit agent" in transcription_lower:
            print("Stop word detected. Exiting program.")
            break

        if not session_active:
            # Look for wake word to start session
            if "hi agent" in transcription_lower:
                user_query = transcription_lower.replace("hi agent", "").strip()
                if not user_query:
                    print("Wake word detected but no query provided. Please speak your query after 'hi agent'.")
                else:
                    session_active = True
                    empty_count = 0  # reset empty count
                    print("Session started. User Query:", user_query)
                    process_query(user_query, ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID)
            else:
                print("Wake word 'hi agent' not detected. Awaiting wake word...")
        else:
            # Session active: process subsequent queries
            if transcription_lower == "":
                empty_count += 1
                print(f"No query detected. Empty count: {empty_count}")
                if empty_count == 3:
                    print("Three successive empty inputs detected. Do you want to end the conversation? (say 'yes' to end)")
                    # Record response for confirmation
                    confirmation_path = record_audio(duration=3, sample_rate=fs) 
                    confirmation = transcribe_audio(confirmation_path, model_size="medium").lower().strip()
                    if "yes" in confirmation:
                        print("Ending session as per user confirmation.")
                        session_active = False
                        empty_count = 0
                        continue
                    else:
                        print("Continuing session.")
                        empty_count = 0  # Reset count if not ending
                if empty_count >= 5:
                    print("Five successive empty inputs detected. Ending session automatically.")
                    session_active = False
                    empty_count = 0
                    continue
            else:
                empty_count = 0  # Reset if input is not empty
                user_query = transcription_lower
                print("New query detected in session:", user_query)
                process_query(user_query, ELEVENLABS_API_KEY, VOICE_ID, MODEL_ID)

        time.sleep(1)

def process_query(user_query: str, api_key: str, voice_id: str, model_id: str):
    input_data = {"question": user_query}
    normalized_input = normalize_qa_input(input_data)
    print("Running RetrievalQA chain with input:", normalized_input)
    raw_answer = qa_chain.invoke(normalized_input)
    # Only use final answer for TTS (ignore thinking)
    _, final_answer = process_answer(raw_answer.get("result", raw_answer))
    print("\n--- Agent Final Answer ---")
    print(final_answer)
    try:
        print("\nConverting text to speech using ElevenLabs and mpv...")
        text_to_speech_mpv(final_answer, api_key, voice_id, model_id)
    except Exception as e:
        print("TTS failed:", e)

# -------------------- Run Chatbot --------------------
if __name__ == "__main__":
    run_chatbot()