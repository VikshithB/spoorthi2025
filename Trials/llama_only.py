import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model loaded.")

# Define sample event-specific documents.
docs = [
    Document(page_content="The keynote speech starts at 10 AM in the main hall."),
    Document(page_content="Registration is open at the front desk."),
    Document(page_content="The event is located in the Conference Center, Room 101."),
    Document(page_content="For any technical issues, contact support at extension 5678.")
]

print("Building FAISS vector store...")
vectorstore = FAISS.from_documents(docs, embeddings)
print("Vector store built.")

def retrieve_context(query, k=2):
    print("Retrieving context for query:", query)
    results = vectorstore.similarity_search(query, k=k)
    print("Retrieved", len(results), "documents.")
    return results

def answer_query(query):
    print("Answering query:", query)
    retrieved_docs = retrieve_context(query)
    
    print("Loading language model (distilgpt2)...")
    model_id = "distilgpt2"  # A smaller model for CPU testing.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Create a HuggingFace text-generation pipeline.
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        truncation=True
    )
    
    # Wrap the pipeline using LangChain's HuggingFacePipeline.
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("Language model loaded.")
    
    # Create a QA chain that uses the retrieved documents.
    chain = load_qa_chain(llm, chain_type="stuff")
    print("Generating answer...")
    answer = chain.run(input_documents=retrieved_docs, question=query)
    return answer

if __name__ == "__main__":
    query = "When does the keynote start?"
    start_time = time.time()
    answer = answer_query(query)
    end_time = time.time()
    print("Answer:", answer)
    print("Total time taken: {:.2f} seconds".format(end_time - start_time))