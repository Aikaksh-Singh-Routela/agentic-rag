# 01_basic_retriever.py - Works with Groq or OpenAI
import os
import sys

# Try to import required modules
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("Installing missing package...")
    os.system("pip install langchain-community -q")
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

# Get API key
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print("=" * 50)
    print("ERROR: API key not found!")
    print("Please set your API key using:")
    print('$env:OPENAI_API_KEY = "your-key-here"')
    print("=" * 50)
    sys.exit(1)

# Check if it's a Groq key (starts with gsk_)
is_groq = api_key.startswith('gsk_')

if is_groq:
    print("⚠️  Detected Groq API key - Note: Groq doesn't support embeddings directly")
    print("   For embeddings, we'll use a free alternative (sentence-transformers)")
    print("   Installing sentence-transformers...")
    os.system("pip install sentence-transformers -q")
    
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Use local embeddings model (free, no API key needed)
    class LocalEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()
        def embed_query(self, text):
            return self.model.encode([text]).tolist()[0]
    
    embeddings = LocalEmbeddings()
    print("✅ Using local embeddings (free)")
else:
    print("✅ Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Sample company policies (our knowledge base)
policies = [
    "Vacation policy: Employees get 20 paid days off per year",
    "Remote work policy: 2 days work from home allowed weekly",
    "Security policy: MFA required for all system access",
    "Expense policy: Meals up to $50 require receipt"
]

print("\n📚 Creating knowledge base...")

# Create vector database
vectorstore = Chroma.from_texts(policies, embeddings)

print(f"✅ Created vector store with {len(policies)} documents\n")

# Test retrieval
query = "How many vacation days do I get?"
print(f"🔍 Searching for: '{query}'\n")

results = vectorstore.similarity_search(query, k=2)

print("📄 Found relevant documents:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")

print("\n✅ Basic retrieval working!")