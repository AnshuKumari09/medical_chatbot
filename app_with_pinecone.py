from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medical-chatbot-secret-2026")

# ============================================================================
# CONFIG
# ============================================================================

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-docs")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

pc = None
pinecone_index = None

# ============================================================================
# PINECONE INITIALIZATION
# ============================================================================

def initialize_pinecone():
    """Connect to existing Pinecone index."""
    global pc, pinecone_index
    
    if not PINECONE_AVAILABLE:
        print("❌ Pinecone not available")
        return False
    
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not set")
            return False
        
        print("🔧 Connecting to Pinecone...")
        
        pc = Pinecone(api_key=api_key)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        stats = pinecone_index.describe_index_stats()
        print(f"✅ Connected: {PINECONE_INDEX_NAME}")
        print(f"   Vectors: {stats.get('total_vector_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def simple_search(query_text, top_k=3):
    """
    Simple keyword-based search.
    NOTE: This won't work perfectly without embeddings,
    but demonstrates the connection.
    """
    
    if not pinecone_index:
        return "Pinecone not connected."
    
    try:
        # Get some random vectors (since we can't embed without the model)
        # This is a workaround - ideally you'd use proper embeddings
        stats = pinecone_index.describe_index_stats()
        total = stats.get('total_vector_count', 0)
        
        if total > 0:
            # Fetch random IDs to demonstrate connection
            results = pinecone_index.query(
                vector=[0.0] * 384,  # Dummy vector
                top_k=min(top_k, 10),
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            
            if matches:
                context = f"Found {len(matches)} documents in Pinecone.\n\n"
                context += "NOTE: Without embeddings, semantic search is disabled.\n"
                context += "Using general medical knowledge instead."
                return context
        
        return "No documents found."
        
    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================================
# LLM
# ============================================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

prompt_template = """You are a medical information assistant.

User question: {question}

Database status: {db_status}

Provide helpful medical information. Always recommend consulting a doctor.
For emergencies, say: "⚠️ CALL 911/108 IMMEDIATELY!"

Keep response under 150 words."""


def get_bot_response(user_question):
    """Get bot response."""
    
    try:
        # Check Pinecone connection
        db_status = simple_search(user_question)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template)
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": user_question,
            "db_status": db_status
        })
        
        return response
        
    except Exception as e:
        return "Error occurred. Please try again."


# ============================================================================
# HELPERS
# ============================================================================

def get_stats():
    """Get stats."""
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            num_vectors = stats.get('total_vector_count', 0)
        except:
            num_vectors = 0
    else:
        num_vectors = 0
    
    return {
        "rag_enabled": pinecone_index is not None,
        "num_vectors": num_vectors,
        "storage": "Pinecone (cloud)",
        "index_name": PINECONE_INDEX_NAME
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form.get("msg", "").strip()
    
    if not user_msg:
        return "Please enter a message", 400
    
    reply = get_bot_response(user_msg)
    return reply


@app.route("/admin")
def admin():
    stats = get_stats()
    
    if not stats['rag_enabled']:
        flash('⚠️ Pinecone not connected', 'warning')
    else:
        flash('⚠️ PDF upload disabled in free tier (memory limit)', 'info')
        flash(f'✅ Connected to Pinecone: {stats["num_vectors"]} vectors', 'success')
    
    return render_template("admin_pinecone.html", stats=stats)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    flash('❌ Upload disabled: Memory limit exceeded', 'error')
    flash('ℹ️ Upload PDFs locally, then push vectors to Pinecone', 'info')
    return redirect(url_for('admin'))


@app.route("/status")
def status():
    stats = get_stats()
    return jsonify({
        "status": "online",
        "pinecone_enabled": PINECONE_AVAILABLE,
        **stats
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


# ============================================================================
# STARTUP
# ============================================================================

def init_app():
    """Initialize."""
    print("\n" + "="*70)
    print("🏥 MEDICAL CHATBOT (Lightweight - No Embeddings)")
    print("="*70)
    
    initialize_pinecone()
    
    stats = get_stats()
    print(f"\n📊 STATUS:")
    print(f"   Connected: {stats['rag_enabled']}")
    print(f"   Vectors: {stats['num_vectors']}")
    
    print("\n⚠️  NOTE: Embeddings disabled to save memory")
    print("   Chatbot uses general medical knowledge")
    print("   For full RAG, upgrade to paid tier or run locally")
    
    print("\n🌐 ENDPOINTS:")
    print("   Chatbot:  /")
    print("   Admin:    /admin")
    print("="*70 + "\n")


# Initialize for Gunicorn
init_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
