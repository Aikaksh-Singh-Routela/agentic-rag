# 02_rag_with_llm.py - Fixed imports for current LangChain
import os
import sys
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# New import paths for current LangChain
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: Set OPENAI_API_KEY environment variable first")
    print('Run: $env:OPENAI_API_KEY = "your-key"')
    sys.exit(1)

print("=" * 60)
print("🤖 Setting up RAG System with Groq LLM")
print("=" * 60)

# Use local embeddings (free)
print("\n📚 Loading local embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Same knowledge base
policies = [
    "Vacation policy: Employees get 20 paid days off per year",
    "Remote work policy: 2 days work from home allowed weekly",
    "Security policy: MFA required for all system access",
    "Expense policy: Meals up to $50 require receipt"
]

# Create vector store
print("📚 Creating knowledge base...")
vectorstore = Chroma.from_texts(policies, embeddings)
print(f"✅ Created vector store with {len(policies)} documents")

# Initialize Groq LLM with CURRENT model
print("\n🚀 Initializing Groq LLM...")
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",  # Current model
    temperature=0,
    max_tokens=500
)

# Create QA chain
print("🔗 Creating QA chain...")
qa_chain = load_qa_chain(llm, chain_type="stuff")

print("\n✅ RAG System Ready!\n")
print("-" * 60)

# Function to answer questions
def answer_question(question):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=2)
    
    if not docs:
        return "No relevant information found."
    
    # Convert to Document objects if needed
    if not isinstance(docs[0], Document):
        docs = [Document(page_content=doc) for doc in docs]
    
    # Generate answer
    response = qa_chain.invoke({"input_documents": docs, "question": question})
    return response["output_text"]

# Test questions
test_questions = [
    "How many vacation days do I get?",
    "What is the remote work policy?",
    "Do I need MFA for system access?",
    "What is the expense policy for meals?"
]

for question in test_questions:
    print(f"\n❓ Question: {question}")
    print("🤔 Thinking...")
    
    try:
        answer = answer_question(question)
        print(f"💬 Answer: {answer}")
        
        # Show source
        docs = vectorstore.similarity_search(question, k=1)
        if docs:
            print(f"📚 Source: {docs[0].page_content}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 60)

print("\n✅ Test complete!")