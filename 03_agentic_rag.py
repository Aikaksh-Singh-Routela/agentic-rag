# 03_agentic_rag.py - Agent decides when to use RAG
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Initialize
api_key = os.getenv('OPENAI_API_KEY')
client = Groq(api_key=api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base
policies = [
    "Vacation policy: Employees get 20 paid days off per year",
    "Remote work policy: 2 days work from home allowed weekly",
    "Security policy: MFA required for all system access",
    "Expense policy: Meals up to $50 require receipt"
]

# Create embeddings
policy_embeddings = model.encode(policies)

def search_policies(query, k=2):
    """Search company policies"""
    query_embedding = model.encode([query])[0]
    similarities = np.dot(policy_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [policies[i] for i in top_indices]

def needs_retrieval(question):
    """Agent decides if question needs policy search"""
    
    # Patterns that DON'T need policy search
    no_retrieval_patterns = [
        r'(hi|hello|hey|greetings)',  # Greetings
        r'(how are you|what\'s up)',   # Casual
        r'(thank|thanks)',             # Gratitude
        r'(weather|news|stock|price)', # External info
        r'(calculate|math|what is \d+.*\d+)'  # Math
    ]
    
    for pattern in no_retrieval_patterns:
        if re.search(pattern, question.lower()):
            return False
    
    # Policy-related keywords that DO need retrieval
    policy_keywords = ['vacation', 'pto', 'remote', 'wfh', 'security', 
                      'mfa', 'expense', 'meal', 'policy', 'hr', 'benefit']
    
    for keyword in policy_keywords:
        if keyword in question.lower():
            return True
    
    # Let LLM decide for ambiguous cases
    prompt = f"""Decide if this question requires searching company policies.
    Answer ONLY "YES" or "NO".
    
    Question: {question}
    
    Does it ask about company policies (vacation, remote work, security, expenses)?
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    
    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"

def answer_without_rag(question):
    """Answer directly without retrieval"""
    prompt = f"""Answer this question directly (no need to search policies):
    
    Question: {question}
    
    Answer naturally and concisely:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    
    return response.choices[0].message.content

def answer_with_rag(question):
    """Answer using retrieved policies"""
    relevant_policies = search_policies(question)
    context = "\n".join(relevant_policies)
    
    prompt = f"""You are an HR assistant. Answer based ONLY on these policies:

POLICIES:
{context}

QUESTION: {question}

ANSWER:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )
    
    return response.choices[0].message.content

def agentic_rag(question):
    """Main agent function - decides what to do"""
    print(f"🤔 Agent thinking about: '{question}'")
    
    if needs_retrieval(question):
        print("📚 Decision: NEEDS policy search")
        return answer_with_rag(question)
    else:
        print("💬 Decision: NO retrieval needed")
        return answer_without_rag(question)

print("=" * 60)
print("🤖 AGENTIC RAG SYSTEM - Decides when to search")
print("=" * 60)

# Test questions - agent will decide for each
test_questions = [
    "Hello!",                                    # No retrieval
    "What's 25 * 4?",                           # No retrieval  
    "How many vacation days do I get?",         # Retrieval needed
    "What's the weather in London?",            # No retrieval
    "Do I need MFA for system access?",         # Retrieval needed
    "Thank you for your help!",                 # No retrieval
    "What is the remote work policy?"           # Retrieval needed
]

for question in test_questions:
    print("\n" + "=" * 60)
    print(f"❓ USER: {question}")
    answer = agentic_rag(question)
    print(f"✅ AGENT: {answer}")
    print("=" * 60)

print("\n🎉 Agentic RAG complete! The agent successfully decided when to search policies.")