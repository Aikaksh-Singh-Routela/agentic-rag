# 04_multi_tool_agent.py - Agent with multiple tools
import os
import re
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

api_key = os.getenv('OPENAI_API_KEY')
client = Groq(api_key=api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base
policies = {
    "vacation": "20 paid days off per year",
    "remote_work": "2 days work from home allowed weekly",
    "security": "MFA required for all system access",
    "expenses": "Meals up to $50 require receipt"
}

policy_list = list(policies.values())
policy_embeddings = model.encode(policy_list)

def search_policies(query):
    """Tool 1: Search company policies"""
    query_embedding = model.encode([query])[0]
    similarities = np.dot(policy_embeddings, query_embedding)
    best_idx = np.argmax(similarities)
    return policy_list[best_idx]

def web_search(query):
    """Tool 2: Search the web (using DuckDuckGo API)"""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('AbstractText'):
            return data['AbstractText'][:300]
        elif data.get('RelatedTopics'):
            first = data['RelatedTopics'][0]
            if isinstance(first, dict) and 'Text' in first:
                return first['Text'][:300]
        return f"Here's what I know about {query}"
    except Exception as e:
        return f"Web search temporarily unavailable"

def calculator(expression):
    """Tool 3: Calculate math expressions"""
    try:
        # Safe evaluation
        allowed = {"abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except:
        return "Invalid calculation"

def decide_tool(question):
    """Agent decides which tool(s) to use"""
    
    # Check for math
    if re.search(r'\d+[\+\-\*\/]\d+', question):
        return "calculator"
    
    # Check for policy keywords
    policy_words = ['vacation', 'pto', 'remote', 'wfh', 'security', 'mfa', 'expense', 'meal']
    if any(word in question.lower() for word in policy_words):
        return "policy"
    
    # Check for web search needs
    web_words = ['weather', 'news', 'current', 'latest', 'today', 'stock', 'price']
    if any(word in question.lower() for word in web_words):
        return "web"
    
    # Let LLM decide for complex cases
    prompt = f"""What tool should answer this question?
    Options: "policy", "web", "calculator", or "direct"
    
    Question: {question}
    
    Answer with just the tool name:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20
    )
    
    return response.choices[0].message.content.strip().lower()

def answer_with_tool(question, tool):
    """Execute the chosen tool"""
    if tool == "policy":
        result = search_policies(question)
        return f"📋 Company policy: {result}"
    
    elif tool == "web":
        result = web_search(question)
        return f"🌐 Web search result: {result}"
    
    elif tool == "calculator":
        result = calculator(question)
        return f"🧮 {result}"
    
    else:  # direct
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content

print("=" * 70)
print("🔧 MULTI-TOOL AGENTIC RAG - Policy + Web + Calculator")
print("=" * 70)

# Test cases
test_questions = [
    "Hello, how are you?",                          # Direct
    "What's 15 * 8?",                               # Calculator
    "How many vacation days do I get?",             # Policy
    "What's the current weather in Tokyo?",         # Web
    "Do I need MFA for system access?",             # Policy
    "What is the capital of France?",               # Web
    "Calculate 144 / 12",                          # Calculator
    "Thank you!",                                    # Direct
]

for question in test_questions:
    print("\n" + "=" * 70)
    print(f"❓ User: {question}")
    print("🤔 Agent deciding which tool to use...")
    
    tool = decide_tool(question)
    print(f"🔧 Selected tool: {tool.upper()}")
    
    answer = answer_with_tool(question, tool)
    print(f"✅ Answer: {answer}")
    print("=" * 70)

print("\n🎉 Multi-tool agent ready! The agent can now:")
print("   📋 Search policies")
print("   🌐 Search the web")
print("   🧮 Calculate math")
print("   💬 Answer directly")