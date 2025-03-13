import os
import warnings
import requests
import json
import whisper

# Suppress FP16 warnings from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ==================== STEP 1: Speech-to-Text with Whisper ====================
def transcribe_audio(audio_file_path, model_size="base"):
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(audio_file_path, task="transcribe")
    return result["text"]

# ==================== STEP 2: Knowledge Base & Vector Store Setup ====================
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Sample documents (replace with your actual event documents)
documents = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is one of the most famous landmarks in the world.",
    "France is known for its art, culture, and cuisine."
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
        # Use streaming so we can iterate over JSON chunks
        response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True)
        response.raise_for_status()
        
        chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode("utf-8"))
                    # Accumulate the response text from each chunk
                    if "response" in obj:
                        chunks.append(obj["response"])
                    if obj.get("done", False):
                        break
                except Exception as e:
                    continue
        return "".join(chunks).strip()

print("Initializing Ollama LLM...")
ollama_llm = OllamaLLM()  # Uses default values defined above
print("Ollama LLM initialized.")

# ==================== STEP 4: Prompt Template & RetrievalQA Chain ====================
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define a prompt template that uses "question" as the variable.
prompt_template = """You are an expert assistant.
Use only the context provided below to answer the question.
If the answer is not contained within the context, say "I don't know."
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

print("Building RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",  # Concatenates the retrieved texts
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
print("RetrievalQA chain built.")

# ==================== Helper: Normalize Input Keys ====================
def normalize_qa_input(input_dict: dict) -> dict:
    """
    Ensure that both "query" and "question" keys are present and have the same value.
    """
    if "query" in input_dict and "question" not in input_dict:
        input_dict["question"] = input_dict["query"]
    elif "question" in input_dict and "query" not in input_dict:
        input_dict["query"] = input_dict["question"]
    elif "query" not in input_dict and "question" not in input_dict:
        raise ValueError("Input must contain either 'query' or 'question'")
    return input_dict

# ==================== STEP 5: Integrated Flow ====================
if __name__ == "__main__":
    # Path to your audio file (ensure the file exists)
    audio_file = "test_recording.wav"
    print("Starting Speech-to-Text...")
    transcription = transcribe_audio(audio_file, model_size="base")
    print("Transcription:", transcription)
    
    # Use the transcription as the user query
    user_query = transcription.strip()
    print("User Query:", user_query)
    
    # Create input dictionary and normalize to include both "query" and "question"
    input_data = {"question": user_query}
    normalized_input = normalize_qa_input(input_data)
    
    print("Running RetrievalQA chain with input:", normalized_input)
    answer = qa_chain.invoke(normalized_input)
    
    print("Final Answer:", answer)