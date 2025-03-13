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
from typing import Tuple
from playsound import playsound

# ElevenLabs imports for real-time streaming TTS
from elevenlabs import stream
from elevenlabs.client import ElevenLabs

# Suppress FP16 warnings from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ==================== STEP 1: Speech-to-Text with Whisper ====================
def transcribe_audio(audio_file_path: str, model_size: str = "base") -> str:
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(audio_file_path, task="transcribe")
    return result["text"]

# ==================== STEP 2: Knowledge Base & Vector Store Setup ====================
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Example documents: a short set plus an essay about India to enrich conversation.
essay_india = (
    "India is a land of diverse cultures, languages, and traditions. From the snow-capped peaks "
    "of the Himalayas to the tropical beaches of Goa, India offers a vast array of experiences and natural beauty. "
    "It is home to a rich history, exemplified by ancient monuments like the Taj Mahal and vibrant traditions that "
    "continue to thrive in modern society. "
    "India's economy is rapidly growing, driven by technology, agriculture, and manufacturing. The country's "
    "cultural heritage and dynamic society have made it a focal point for international tourism and global business. "
    "The blend of traditional customs and modern innovation makes India a unique and inspiring nation."
)

documents = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is one of the most famous landmarks in the world.",
    "France is known for its art, culture, and cuisine.",
    essay_india
]

print("Embedding documents...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(documents, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store and retriever initialized.")

# ==================== STEP 3: Custom Ollama LLM Wrapper ====================
from langchain.llms.base import LLM
from pydantic import Field

class OllamaLLM(LLM):
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:1.5b"
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
ollama_llm = OllamaLLM()  # Uses default values defined above
print("Ollama LLM initialized.")

# ==================== STEP 4: Prompt Template & RetrievalQA Chain ====================
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """You are an expert assistant.
Use only the context provided below to answer the question.
If the answer is not contained within the context, say "I don't know."
"Answer in 2-3 sentences only. Be concise and to the point."
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

print("Building RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",  # Concatenates retrieved texts
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)
print("RetrievalQA chain built.")

# ==================== Helper: Normalize Input Keys ====================
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

# ==================== Helper: Process Answer ====================
def process_answer(answer_text: str) -> Tuple[str, str]:
    """
    Extract the 'thinking' part from within <think>...</think> tags and separate the final answer.
    If the final answer is insufficient, ask for clarification.
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

# ==================== STEP 5: Continuous Integrated Flow with Wake and Stop Words ====================
if __name__ == "__main__":
    # Set your ElevenLabs API key and configuration (replace with your own values)
    ELEVENLABS_API_KEY = "sk_d897d0a1131929c79ef50d9a27aa308d6aded0c429ccaef2"
    VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # You can choose any valid voice ID from your list
    MODEL_ID = "eleven_multilingual_v2"  # For example
    
    client = ElevenLabs()  # Initialize ElevenLabs client

    print("Starting continuous listening for wake word...")
    fs = 16000  # Sample rate for recording
    duration = 10  # Record 10-second snippets

    while True:
        print("\nListening for wake word 'hi agent'...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to finish
        wav.write("temp.wav", fs, recording)
        
        transcription = transcribe_audio("temp.wav", model_size="base")
        print("Transcription:", transcription)
        
        transcription_lower = transcription.lower()
        # Check for the stop word first
        if "bye agent" in transcription_lower:
            print("Stop word detected. Exiting the program.")
            break
        
        # Check for the wake word "hi agent"
        if "hi agent" in transcription_lower:
            user_query = transcription_lower.replace("hi agent", "").strip()
            if not user_query:
                print("Wake word detected but no question provided. Please state your question after 'hi agent'.")
            else:
                print("Wake word detected. User Query:", user_query)
                input_data = {"question": user_query}
                normalized_input = normalize_qa_input(input_data)
                print("Running RetrievalQA chain with input:", normalized_input)
                raw_answer = qa_chain.invoke(normalized_input)
                thinking, final_answer = process_answer(raw_answer.get("result", raw_answer))
                print("\n--- Agent Thinking ---")
                print(thinking)
                print("\n--- Agent Final Answer ---")
                print(final_answer)
                
                # Convert the final answer to speech using ElevenLabs streaming TTS
                try:
                    print("\nConverting text to speech using ElevenLabs...")
                    audio_stream = client.text_to_speech.convert_as_stream(
                        text=final_answer,
                        voice_id=VOICE_ID,
                        model_id=MODEL_ID
                    )
                    print("Playing audio...")
                    stream(audio_stream)
                except Exception as e:
                    print("TTS failed:", e)
        else:
            print("Wake word 'hi agent' not detected.")
        
        # Short pause before next iteration
        time.sleep(1)
