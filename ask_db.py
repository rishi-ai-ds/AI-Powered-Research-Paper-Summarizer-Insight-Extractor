from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from  gemini_file import *

# load embedding model 
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


# load FAISS vector store 

vector_db = FAISS.load_local(
    "research_papers_faiss",
    embeddings,
    allow_dangerous_deserialization= True
)

print(f"vector loaded: {vector_db.index.ntotal}")

# user query 
# user_query ="Which papers discuss privacy in ML"
user_query="which paper use entropy-based evaluation"

# similarity search , Keyword search, hybrid search

# retrival 
top_k =3
results = vector_db.similarity_search(user_query, k=top_k)

# display results 
print("\n Top Relevent Documents:\n")

content=""
for idx, doc in enumerate(results, 1):
    
    print(f"{idx, {doc.page_content}}")
    content+=", "+ doc.page_content



print("-------- Gemini Response-------------")
ask_gemini(content, user_query)
































# https://chatgpt.com/c/69a86e7d-3bb8-8324-9eaf-c5807d399382