# 02_rag_simple.py - Simple RAG with direct Groq API
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

api_key = os.getenv('OPENAI_API_KEY')
client = Groq(api_key=api_key)

# Load embedding model
print("📚 Loading embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base
policies = [
    "Vacation policy: Employees get 20 paid days off per year",
    "Remote work policy: 2 days work from home allowed weekly",
    "Security policy: MFA required for all system access",
    "Expense policy: Meals up to $50 require receipt"
]

# Create embeddings for all policies
print("📚 Creating vector embeddings...")
policy_embeddings = model.encode(policies)

def search_policies(query, k=2):
    """Find most relevant policies"""
    query_embedding = model.encode([query])[0]
    
    # Calculate similarities
    similarities = np.dot(policy_embeddings, query_embedding)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    return [policies[i] for i in top_indices]

def answer_question(question):
    """Generate answer using Groq"""
    # Retrieve relevant policies
    relevant_policies = search_policies(question)
    
    # Build prompt with context
    context = "\n".join(relevant_policies)
    prompt = f"""You are a helpful HR assistant. Answer questions based ONLY on the following company policies:

POLICIES:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Get response from Groq
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    
    return completion.choices[0].message.content

print("=" * 60)
print("🤖 RAG System Ready (Simple Version)")
print("=" * 60)

# Test
questions = [
    "How many vacation days?",
    "Can I work from home?",
    "Do I need MFA?"
]

for q in questions:
    print(f"\n❓ {q}")
    print(f"💬 {answer_question(q)}")
    print("-" * 40)