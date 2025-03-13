import os
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ----- Step 1: Set up the vector store using FAISS for our documents -----
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Example documents (replace with your actual documents)
documents = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is one of the most famous landmarks in the world.",
    "France is known for its art, culture, and cuisine."
]

print("Embedding documents...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(documents, embedding_model)
# Set up the retriever to return the top 3 most similar documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store and retriever initialized.")

# ----- Step 2: Load the transformer model directly using the pipeline API -----
from transformers import pipeline

print("Loading transformer model Janus-Pro-1B from transformers...")
# device=-1 forces CPU; change to device=0 if you have a GPU and proper CUDA setup
generator_pipeline = pipeline("text-generation", model="deepseek-ai/Janus-Pro-1B", device=-1)
print("Transformer model loaded.")

# Wrap the transformers pipeline with LangChain's HuggingFacePipeline wrapper
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# ----- Step 3: Define a prompt template for question-answering -----
from langchain.prompts import PromptTemplate

prompt_template = """You are an expert assistant.
Use only the context provided below to answer the question.
If the answer is not contained within the context, say "I don't know."
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# ----- Step 4: Create a RetrievalQA chain using the retriever and the transformer LLM -----
from langchain.chains import RetrievalQA

print("Building RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # This method concatenates the retrieved texts
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
print("RetrievalQA chain built.")

# ----- Step 5: Run the RAG pipeline with an example query -----
if __name__ == "__main__":
    query = "What is the capital of France?"
    print("Running query:", query)
    # Ensure the prompt input key "question" is used
    answer = qa_chain.invoke({"question": query})
    print("Question:", query)
    print("Answer:", answer)