from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not installed. Install: pip install pinecone-client langchain-pinecone")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_FOLDER = "uploaded_pdfs"  # Temporary storage (ephemeral)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {'pdf'}
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-docs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

vectorstore = None
retriever = None
embeddings = None
pinecone_index = None

# ============================================================================
# INITIALIZE PINECONE
# ============================================================================

def initialize_pinecone():
    """Initialize Pinecone vector store."""
    global vectorstore, retriever, embeddings, pinecone_index
    
    if not PINECONE_AVAILABLE:
        print("❌ Pinecone not available")
        return False
    
    try:
        # Get API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not set")
            return False
        
        print("🔧 Initializing Pinecone...")
        
        # Initialize embeddings
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"📝 Creating new index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Get index
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create vector store
        vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        print(f"✅ Pinecone initialized: {PINECONE_INDEX_NAME}")
        
        # Get stats
        stats = pinecone_index.describe_index_stats()
        print(f"   Vectors in index: {stats.get('total_vector_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
        return False


def add_pdf_to_vectorstore(pdf_path):
    """Add PDF to Pinecone vector store."""
    global vectorstore
    
    try:
        print(f"\n📄 Processing: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"   ✅ Loaded {len(docs)} pages")
        
        if not docs:
            return False, "PDF is empty or unreadable", 0
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata['source'] = Path(pdf_path).name
        
        # Add to Pinecone
        vectorstore.add_documents(chunks)
        print(f"   ✅ Added to Pinecone")
        
        return True, f"Added {len(chunks)} chunks", len(chunks)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e), 0


# ============================================================================
# LLM SETUP
# ============================================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

rag_prompt_template = """You are a medical information assistant with access to medical documents.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

🚨 CRITICAL RULES:
1. EMERGENCIES: Chest pain, breathing difficulty, severe bleeding 
   → "⚠️ MEDICAL EMERGENCY! Call 911/108 IMMEDIATELY!"

2. BOUNDARIES: You CANNOT diagnose or prescribe
   → Always say "Consult a doctor for diagnosis/prescription"

3. ICD CODES: E11.9=Type 2 Diabetes, I10=Hypertension, J45.9=Asthma, etc.

4. LANGUAGE: Support English and Hindi

RESPONSE GUIDELINES:
- Use CONTEXT if relevant
- Be concise (<150 words)
- Always recommend doctor consultation
- Cite documents when using their info

Your answer:"""


def get_bot_response(user_question):
    """Get chatbot response with RAG."""
    
    try:
        # Retrieve context
        if retriever and vectorstore:
            docs = retriever.invoke(user_question)
            
            if docs:
                context = "\n\n".join([
                    f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                    for doc in docs
                ])
            else:
                context = "No relevant documents found."
        else:
            context = "Vector store not initialized."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompt_template)
        ])
        
        # Generate response
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": user_question
        })
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Sorry, I encountered an error. Please try again."


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_stats():
    """Get system statistics."""
    if vectorstore and pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            num_vectors = stats.get('total_vector_count', 0)
        except:
            num_vectors = 0
    else:
        num_vectors = 0
    
    return {
        "rag_enabled": vectorstore is not None,
        "num_vectors": num_vectors,
        "storage": "Pinecone (persistent)",
        "index_name": PINECONE_INDEX_NAME
    }


# ============================================================================
# ROUTES - CHATBOT
# ============================================================================

@app.route("/")
def home():
    """Main chatbot interface."""
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    """Chat endpoint."""
    user_msg = request.form.get("msg", "").strip()
    
    if not user_msg:
        return "Please enter a message", 400
    
    try:
        reply = get_bot_response(user_msg)
        return reply
    except Exception as e:
        return f"Error: {str(e)}", 500


# ============================================================================
# ROUTES - ADMIN PANEL
# ============================================================================

@app.route("/admin")
def admin():
    """Admin panel."""
    stats = get_stats()
    
    if not stats['rag_enabled']:
        flash('⚠️ Pinecone not configured. Set PINECONE_API_KEY in environment variables.', 'warning')
    
    return render_template("admin_pinecone.html", stats=stats)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload."""
    
    if not vectorstore:
        flash('❌ Vector store not initialized', 'error')
        return redirect(url_for('admin'))
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('admin'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('admin'))
    
    if not allowed_file(file.filename):
        flash('Only PDF files allowed', 'error')
        return redirect(url_for('admin'))
    
    try:
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add to Pinecone
        success, message, num_chunks = add_pdf_to_vectorstore(filepath)
        
        # Delete temp file
        os.remove(filepath)
        
        if success:
            flash(f'✅ Successfully added "{filename}" to Pinecone ({num_chunks} chunks)', 'success')
        else:
            flash(f'❌ Error: {message}', 'error')
            
    except Exception as e:
        flash(f'❌ Upload failed: {str(e)}', 'error')
    
    return redirect(url_for('admin'))


@app.route("/status")
def status():
    """API endpoint for system status."""
    stats = get_stats()
    
    return jsonify({
        "status": "online",
        "pinecone_enabled": PINECONE_AVAILABLE,
        **stats
    })


@app.route("/health")
def health():
    """Health check."""
    return jsonify({"status": "healthy"})


# ============================================================================
# STARTUP
# ============================================================================

def init_app():
    """Initialize app."""
    print("\n" + "="*70)
    print("🏥 MEDICAL CHATBOT WITH PINECONE")
    print("="*70)
    
    # Initialize Pinecone
    pinecone_ready = initialize_pinecone()
    
    stats = get_stats()
    print(f"\n📊 STATUS:")
    print(f"   RAG Enabled: {stats['rag_enabled']}")
    print(f"   Vectors: {stats['num_vectors']}")
    print(f"   Storage: {stats['storage']}")
    
    print("\n🌐 ENDPOINTS:")
    print("   Chatbot:  /")
    print("   Admin:    /admin")
    print("   Status:   /status")
    print("="*70 + "\n")


# ============================================================================
# RUN - IMPORTANT: Initialize at module level for Gunicorn
# ============================================================================

# Initialize app when module is imported (needed for Gunicorn)
init_app()

if __name__ == "__main__":
    # This only runs when using `python app_with_pinecone.py` directly
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
