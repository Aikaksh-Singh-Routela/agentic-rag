# test_groq.py - Updated with current Groq models
import os
from groq import Groq

api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key found: {api_key[:20]}...")

client = Groq(api_key=api_key)

# Try one of these current models:
# - "llama-3.3-70b-versatile" (most capable)
# - "llama-3.1-8b-instant" (fastest, free tier)
# - "mixtral-8x7b-32768" is DECOMMISSIONED

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # Changed from decommissioned model
    messages=[{"role": "user", "content": "Say 'Hello, RAG system is working!'"}],
    temperature=0
)

print(completion.choices[0].message.content)