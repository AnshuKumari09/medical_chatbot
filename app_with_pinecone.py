from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
from pathlib import Path
import time  # ADDED for rate limiting

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask_cors import CORS
# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not installed")

load_dotenv()

app = Flask(__name__)
CORS(app) 
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_FOLDER = "uploaded_pdfs"
ALLOWED_EXTENSIONS = {'pdf'}
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-docs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

pc = None
pinecone_index = None

# ============================================================================
# INITIALIZE PINECONE (WITHOUT HEAVY EMBEDDINGS)
# ============================================================================

def initialize_pinecone():
    """Initialize Pinecone with inference API (no local embeddings)."""
    global pc, pinecone_index
    
    if not PINECONE_AVAILABLE:
        print("❌ Pinecone not available")
        return False
    
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not set")
            return False
        
        print("🔧 Initializing Pinecone (lightweight mode)...")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"📝 Creating new index with Pinecone embeddings: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,  # multilingual-e5-large dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Get index
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        print(f"✅ Pinecone initialized: {PINECONE_INDEX_NAME}")
        
        # Get stats
        stats = pinecone_index.describe_index_stats()
        print(f"   Vectors in index: {stats.get('total_vector_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
        return False


def add_pdf_to_vectorstore(pdf_path):
    """Add PDF to Pinecone using inference API with BATCH PROCESSING + RATE LIMITING."""
    global pinecone_index
    
    if not pinecone_index:
        return False, "Pinecone not initialized", 0
    
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
        
        # SMALLER batches + Rate limiting to respect Pinecone free tier limits
        EMBED_BATCH_SIZE = 20  # Reduced from 50 to 20 (slower but safer)
        UPSERT_BATCH_SIZE = 100
        SLEEP_BETWEEN_BATCHES = 3  # 3 seconds between embedding batches
        
        all_vectors = []
        total_embedded = 0
        failed_batches = 0
        
        # Process chunks in batches for embedding
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch_chunks = chunks[i:i+EMBED_BATCH_SIZE]
            
            # Prepare texts and metadata for this batch
            texts = []
            metadatas = []
            for j, chunk in enumerate(batch_chunks):
                texts.append(chunk.page_content)
                metadatas.append({
                    "text": chunk.page_content,
                    "source": Path(pdf_path).name,
                    "chunk_id": i + j
                })
            
            # Embed this batch using Pinecone inference
            try:
                embeddings = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=texts,
                    parameters={"input_type": "passage"}
                )
                
                # Prepare vectors
                for j, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                    all_vectors.append({
                        "id": f"{Path(pdf_path).stem}_{i+j}",
                        "values": embedding['values'],
                        "metadata": metadata
                    })
                
                total_embedded += len(texts)
                print(f"   ⏳ Embedded {total_embedded}/{len(chunks)} chunks...")
                
                # CRITICAL: Sleep between batches to respect rate limits
                if i + EMBED_BATCH_SIZE < len(chunks):  # Don't sleep after last batch
                    time.sleep(SLEEP_BETWEEN_BATCHES)
                
            except Exception as e:
                error_msg = str(e)
                failed_batches += 1
                
                # If rate limited, wait longer
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"   ⚠️ Rate limit hit at batch {i//EMBED_BATCH_SIZE + 1}, waiting 10 seconds...")
                    time.sleep(10)
                    
                    # Retry this batch once
                    try:
                        embeddings = pc.inference.embed(
                            model="multilingual-e5-large",
                            inputs=texts,
                            parameters={"input_type": "passage"}
                        )
                        
                        for j, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                            all_vectors.append({
                                "id": f"{Path(pdf_path).stem}_{i+j}",
                                "values": embedding['values'],
                                "metadata": metadata
                            })
                        
                        total_embedded += len(texts)
                        print(f"   ✅ Retry successful! Embedded {total_embedded}/{len(chunks)}")
                        time.sleep(SLEEP_BETWEEN_BATCHES)
                        
                    except Exception as retry_error:
                        print(f"   ❌ Retry failed for batch {i//EMBED_BATCH_SIZE + 1}: {retry_error}")
                        continue
                else:
                    print(f"   ⚠️ Batch {i//EMBED_BATCH_SIZE + 1} failed: {e}")
                    continue
        
        print(f"   📊 Embedding complete: {total_embedded} succeeded, {failed_batches} batches failed")
        
        # Upsert vectors in batches
        total_upserted = 0
        for i in range(0, len(all_vectors), UPSERT_BATCH_SIZE):
            batch = all_vectors[i:i+UPSERT_BATCH_SIZE]
            try:
                pinecone_index.upsert(vectors=batch)
                total_upserted += len(batch)
                print(f"   ⏳ Upserted {total_upserted}/{len(all_vectors)} vectors...")
                
                # Small sleep between upserts too
                if i + UPSERT_BATCH_SIZE < len(all_vectors):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"   ⚠️ Upsert batch failed: {e}")
                continue
        
        print(f"   ✅ Successfully added {total_upserted} vectors to Pinecone")
        
        return True, f"Added {total_upserted} chunks", total_upserted
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e), 0


def query_pinecone(query_text, top_k=3):
    """Query Pinecone using inference API."""
    global pinecone_index
    
    if not pinecone_index:
        return []
    
    try:
        # Embed query using Pinecone inference
        query_embedding = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query"}
        )
        
        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding[0]['values'],
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract documents
        docs = []
        for match in results.get('matches', []):
            if match.get('metadata'):
                docs.append({
                    'content': match['metadata'].get('text', ''),
                    'source': match['metadata'].get('source', 'unknown'),
                    'score': match.get('score', 0)
                })
        
        return docs
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []


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
        # Retrieve context from Pinecone
        if pinecone_index:
            docs = query_pinecone(user_question)
            
            if docs:
                context = "\n\n".join([
                    f"[Source: {doc['source']}]\n{doc['content']}"
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
        "storage": "Pinecone Inference (e5-large/1024d)",
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
    
    if not pinecone_index:
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
        
        # Add to Pinecone (will take longer due to rate limiting)
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
    print("🏥 MEDICAL CHATBOT WITH PINECONE (OPTIMIZED)")
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
# RUN
# ============================================================================

# Initialize app when module is imported (needed for Gunicorn)
init_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)



