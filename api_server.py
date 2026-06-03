# api_server.py - Flask API for your Agentic RAG
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Initialize
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
    query_embedding = model.encode([query])[0]
    similarities = np.dot(policy_embeddings, query_embedding)
    best_idx = np.argmax(similarities)
    return policy_list[best_idx]

def web_search(query):
    try:
        import requests
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('AbstractText'):
            return data['AbstractText'][:300]
        return f"Information about {query}"
    except:
        return "Web search unavailable"

def calculator(expression):
    try:
        allowed = {"abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except:
        return "Invalid calculation"

def decide_tool(question):
    if re.search(r'\d+[\+\-\*\/]\d+', question):
        return "calculator"
    policy_words = ['vacation', 'pto', 'remote', 'wfh', 'security', 'mfa', 'expense']
    if any(word in question.lower() for word in policy_words):
        return "policy"
    web_words = ['weather', 'news', 'current', 'latest', 'today', 'capital']
    if any(word in question.lower() for word in web_words):
        return "web"
    return "direct"

def answer_with_tool(question, tool):
    if tool == "policy":
        return f"📋 {search_policies(question)}"
    elif tool == "web":
        return f"🌐 {web_search(question)}"
    elif tool == "calculator":
        return f"🧮 {calculator(question)}"
    else:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    tool = decide_tool(question)
    answer = answer_with_tool(question, tool)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'tool_used': tool
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Agentic RAG API Starting...")
    print("📍 Health check: http://localhost:8080/health")
    print("📍 Ask endpoint: POST http://localhost:8080/ask")
    app.run(host='0.0.0.0', port=8080, debug=True)