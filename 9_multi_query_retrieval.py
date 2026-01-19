from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
from google import genai
import os
import json

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
embedding_model = VoyageAIEmbeddings(model="voyage-4")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations using Gemini
# ──────────────────────────────────────────────────────────────────

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return ONLY a JSON array with 3 alternative queries. Example format:
["query 1", "query 2", "query 3"]"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# Parse JSON response
response_text = response.text.strip()
# Remove markdown code blocks if present
if response_text.startswith("```"):
    response_text = response_text.split("\n", 1)[1].rsplit("\n", 1)[0]
query_variations = json.loads(response_text)

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")


# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results  
#     [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
# ]