from dotenv import load_dotenv
from langchain_chroma import Chroma
from google import genai
from google.genai import types
from langchain_voyageai import VoyageAIEmbeddings
import os

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = VoyageAIEmbeddings(model="voyage-4")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Store our conversation history
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Build context from chat history
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
            for msg in chat_history
        ])
        
        rewrite_prompt = f"""Given this conversation history:
{history_text}

Rewrite this new question to be standalone and searchable. Just return the rewritten question, nothing else.
New question: {user_question}"""
        
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=rewrite_prompt
        )
        search_question = result.text.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt with history
    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n" + "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
            for msg in chat_history
        ]) + "\n\n"
    
    combined_input = f"""{history_text}Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    # Step 4: Get the answer using Gemini
    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=combined_input,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant that answers questions based on provided documents and conversation history."
        )
    )
    answer = result.text
    
    # Step 5: Remember this conversation
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()