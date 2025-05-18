import os
import json
import logging
import time
import threading
from functools import wraps

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# BGE M3 specific imports
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Backend specific imports
import google.generativeai as genai
from utils import generate_similarity_paragraph_stream # Assuming utils.py is in the same directory

# Load environment variables from .env file
load_dotenv()

# Configure logging (from bge_m3_api)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Gemini API Configuration (from backend) ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    logger.info("Gemini API configured.")
else:
    logger.warning("GEMINI_API_KEY not found in environment. Gemini related endpoints will likely fail.")

# --- BGE M3 Model Globals and Initialization (from bge_m3_api) ---
MODEL = None
MODEL_LOADED = threading.Event()
COLBERT_VEC_CACHE = {} # Cache for Colbert vectors {text_string: colbert_vector}

def initialize_bge_m3_model(model_name='BAAI/bge-m3', use_fp16=None, device=None):
    global MODEL
    if device is None:
        import torch # Import torch here to avoid making it a hard dependency if only backend is used.
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if use_fp16 is None:
        use_fp16 = device.startswith('cuda')
    
    logger.info(f"Initializing BGE-M3 model ({model_name}) on device: {device} with FP16: {use_fp16}")
    start_time = time.time()
    try:
        MODEL = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        load_time = time.time() - start_time
        logger.info(f"BGE-M3 Model initialization complete in {load_time:.2f} seconds")
        MODEL_LOADED.set()
    except Exception as e:
        logger.error(f"BGE-M3 Model initialization failed: {str(e)}")
        # Optionally, re-raise or handle to prevent app start if model is critical for all ops
        raise

def start_model_initialization_thread():
    def initialize_model_task():
        try:
            model_name_to_load = os.environ.get('MODEL_NAME', 'BAAI/bge-m3')
            logger.info(f"Attempting to load BGE-M3 model using identifier: {model_name_to_load}")
            if os.path.isdir(model_name_to_load):
                logger.info(f"Treating BGE-M3 model identifier as a local path: {model_name_to_load}")
                if not os.path.exists(os.path.join(model_name_to_load, 'config.json')):
                    logger.warning(f"Local BGE-M3 model path {model_name_to_load} exists, but config.json is missing.")
            else:
                logger.info(f"Treating BGE-M3 model identifier '{model_name_to_load}' as a Hugging Face Hub ID.")
            
            use_fp16_str = os.environ.get('USE_FP16', '').lower()
            use_fp16 = None
            if use_fp16_str in ('true', 'false'):
                use_fp16 = use_fp16_str == 'true'
            device = os.environ.get('MODEL_DEVICE', None)
            
            initialize_bge_m3_model(model_name=model_name_to_load, use_fp16=use_fp16, device=device)
        except Exception as e:
            logger.error(f"Background BGE-M3 model initialization failed: {str(e)}")
            # Consider how to handle this - app might be partially functional or should fail.
            # For now, it logs the error, and /health will show model_loaded: false.

    thread = threading.Thread(target=initialize_model_task)
    thread.daemon = True
    thread.start()

# --- API Key Authentication (from bge_m3_api) ---
_raw_api_keys_env = os.environ.get('ALLOWED_API_KEYS')
API_KEYS = set()
if _raw_api_keys_env:
    API_KEYS = set(key.strip() for key in _raw_api_keys_env.split(',') if key.strip())
logger.info(f"API Keys loaded: {API_KEYS if API_KEYS else 'No API keys configured (open access for BGE M3 endpoints)'}")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not API_KEYS or '*' in API_KEYS: # Open access if no keys defined or '*' is a key
            return f(*args, **kwargs)
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
            logger.warning(f"Access denied: Invalid or missing API key for BGE M3 endpoint. Provided: '{api_key}'")
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

def wait_for_model(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not MODEL_LOADED.is_set():
            # Wait for a short period in case the model is just about to be ready
            MODEL_LOADED.wait(timeout=5) # Wait up to 5 seconds
            if not MODEL_LOADED.is_set():
                 logger.error("Attempted to access BGE M3 model but it's not loaded.")
                 return jsonify({"error": "BGE M3 Model is still loading or failed to load. Please try again later."}), 503
        if MODEL is None: # Should be redundant if MODEL_LOADED.is_set() is true and init logic is correct
            logger.error("MODEL_LOADED event is set, but MODEL global is None. This indicates an issue in initialization logic.")
            return jsonify({"error": "BGE M3 Model inconsistency. Please check logs."}), 500
        return f(*args, **kwargs)
    return decorated_function

# --- Routes from backend/app.py ---
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Combined API"})

@app.route('/api/hello')
def hello():
    return jsonify({"message": "Hello from the backend component!"})

@app.route('/gemini/test')
def gemini_test():
    if not gemini_api_key:
        return jsonify({"status": "error", "message": "Gemini API key not configured"}), 500
    try:
        model = genai.GenerativeModel('gemini-pro') # Changed from 'models/gemini-2.0-flash' as it might not be a valid public model id
        response = model.generate_content('Who came first, the chicken or the egg?')
        return jsonify({"status": "success", "response": response.text})
    except Exception as e:
        logger.error(f"Error in /gemini/test: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/compare-papers', methods=['POST'])
def compare_papers():
    if not gemini_api_key:
        return jsonify({"status": "error", "message": "Gemini API key not configured"}), 500
        
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"status": "error", "message": "Two PDF files required"}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    if not (file1.filename.endswith('.pdf') and file2.filename.endswith('.pdf')):
        return jsonify({"status": "error", "message": "Both files must be PDFs"}), 400
    
    try:
        pdf_bytes1 = file1.read()
        pdf_bytes2 = file2.read()
        
        # Use a closure for the stream generator
        def stream_generator():
            yield json.dumps({"status": "processing", "message": "Starting comparison..."}) + '\\n'
            try:
                for chunk in generate_similarity_paragraph_stream(pdf_bytes1, pdf_bytes2):
                    yield json.dumps({"status": "generating", "chunk": chunk}) + '\\n'
                yield json.dumps({"status": "complete", "paper1_name": file1.filename, "paper2_name": file2.filename}) + '\\n'
            except Exception as e_stream:
                 logger.error(f"Error during streaming in compare_papers: {str(e_stream)}")
                 # It's tricky to yield an error once streaming has started with event-stream. Client needs to handle abrupt end or last message.
                 # This error will be logged. Consider how to signal this to client if necessary.
                 # For now, just end the stream. A more robust solution might involve a specific error event type.
                 yield json.dumps({"status": "error_streaming", "message": str(e_stream)}) + '\\n'


        return Response(stream_with_context(stream_generator()), mimetype='application/json-seq') # Using application/json-seq or text/event-stream
    except Exception as e:
        logger.error(f"Error in /api/compare-papers: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Routes from bge_m3_api/app.py ---
@app.route('/health', methods=['GET'])
def health_check():
    bge_model_status = "loaded" if MODEL_LOADED.is_set() and MODEL is not None else "not_loaded"
    if MODEL_LOADED.is_set() and MODEL is None: # Check for inconsistency
        bge_model_status = "error_inconsistent_state"
    
    gemini_configured = bool(gemini_api_key)
    
    health_status = {
        "status": "healthy", # Overall service status
        "bge_m3_model_status": bge_model_status,
        "gemini_api_configured": gemini_configured,
        "timestamp": time.time()
    }
    
    # Determine overall HTTP status code
    # If model is critical and not loaded, service might be considered unhealthy.
    # For now, returns 200 but indicates model status.
    # Cloud Run health checks typically look for 200.
    # If BGE model fails to load, this endpoint will reflect that.
    http_status_code = 200 
    if not MODEL_LOADED.is_set() and os.environ.get('REQUIRE_BGE_MODEL_FOR_HEALTH', 'false').lower() == 'true':
        # If an env var dictates BGE model is essential for health, then return 503.
        http_status_code = 503
        health_status["status"] = "unhealthy_bge_model_not_ready"

    return jsonify(health_status), http_status_code


@app.route('/encode', methods=['POST'])
@require_api_key
@wait_for_model
def encode_texts():
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' in request body"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "'texts' must be a list of strings"}), 400
        if not texts: # If list is empty
            return jsonify({"dense_vecs": [], "lexical_weights": [], "colbert_vecs": []}), 200

        params = {
            'return_dense': data.get('return_dense', True),
            'return_sparse': data.get('return_sparse', False),
            'return_colbert_vecs': data.get('return_colbert_vecs', False),
            'batch_size': data.get('batch_size', 32), # Default from BGE M3, check if FlagEmbedding uses it
            'max_length': data.get('max_length', 8192) # Default from BGE M3
        }
        
        logger.debug(f"Encoding texts with params: {params}")
        
        final_embeddings_dict = {}

        # Colbert Vector Caching Logic for /encode
        if params['return_colbert_vecs']:
            cached_colberts_for_batch = []
            all_colberts_in_cache_for_batch = True
            texts_needing_fresh_colbert = []
            indices_for_fresh_colbert = []
            
            temp_colbert_results = [None] * len(texts)

            for i, text_input in enumerate(texts):
                if text_input in COLBERT_VEC_CACHE:
                    temp_colbert_results[i] = COLBERT_VEC_CACHE[text_input]
                else:
                    all_colberts_in_cache_for_batch = False
                    texts_needing_fresh_colbert.append(text_input)
                    indices_for_fresh_colbert.append(i)
            
            if all_colberts_in_cache_for_batch:
                logger.info(f"All {len(texts)} Colbert vectors for /encode batch found in cache.")
                final_embeddings_dict['colbert_vecs'] = temp_colbert_results # Use the ordered list from cache
            elif texts_needing_fresh_colbert:
                logger.info(f"Encoding {len(texts_needing_fresh_colbert)} texts freshly for Colbert vectors in /encode.")
                # Encode only the texts not found in cache to get their Colbert vectors
                fresh_colbert_output = MODEL.encode(
                    texts_needing_fresh_colbert,
                    return_dense=False, return_sparse=False, return_colbert_vecs=True
                )
                if 'colbert_vecs' in fresh_colbert_output and fresh_colbert_output['colbert_vecs'] is not None:
                    for i, fresh_vec in enumerate(fresh_colbert_output['colbert_vecs']):
                        original_idx = indices_for_fresh_colbert[i]
                        text_val = texts_needing_fresh_colbert[i]
                        temp_colbert_results[original_idx] = fresh_vec # Place in correct original position
                        COLBERT_VEC_CACHE[text_val] = fresh_vec # Cache it
                final_embeddings_dict['colbert_vecs'] = temp_colbert_results


            # Now, if other embeddings are needed, encode all texts, but don't re-request colbert if we handled it
            if params['return_dense'] or params['return_sparse']:
                params_for_others = params.copy()
                params_for_others['return_colbert_vecs'] = False # We've handled Colbert
                other_model_output = MODEL.encode(texts, **params_for_others)
                final_embeddings_dict.update(other_model_output)
        else: # Colbert not requested at all
            if params['return_dense'] or params['return_sparse']:
                model_output = MODEL.encode(texts, **params)
                final_embeddings_dict.update(model_output)

        result = {}
        if 'dense_vecs' in final_embeddings_dict and final_embeddings_dict['dense_vecs'] is not None:
            result['dense_vecs'] = final_embeddings_dict['dense_vecs'].tolist()
        if 'lexical_weights' in final_embeddings_dict and final_embeddings_dict['lexical_weights'] is not None:
            result['lexical_weights'] = final_embeddings_dict['lexical_weights']
        if 'colbert_vecs' in final_embeddings_dict and final_embeddings_dict['colbert_vecs'] is not None:
            result['colbert_vecs'] = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in final_embeddings_dict['colbert_vecs'] if vec is not None]
        
        if data.get('compute_colbert_pairwise_scores', False) and params['return_colbert_vecs'] and \
           'colbert_vecs' in result and result['colbert_vecs'] and len(texts) > 1:
            
            # Ensure we have the numpy arrays for scoring from the final_embeddings_dict or cache
            colbert_vectors_for_scoring = []
            for text_input in texts: # Get them in order, preferably from cache if already np.ndarray
                 if text_input in COLBERT_VEC_CACHE: # COLBERT_VEC_CACHE stores np.ndarray
                     colbert_vectors_for_scoring.append(COLBERT_VEC_CACHE[text_input])
                 # Fallback if somehow not populated in cache (should not happen with current logic)
                 # This part might be complex if vectors were not fetched earlier
                 # For simplicity, this example assumes colbert_vecs in result are sufficient or cache is prime
            
            # Fallback if colbert_vectors_for_scoring isn't populated correctly (e.g. cache issue)
            # This indicates a potential logic flaw above if this branch is hit often.
            if not colbert_vectors_for_scoring or len(colbert_vectors_for_scoring) != len(texts):
                 logger.warning("Re-fetching colbert vectors for pairwise scoring due to inconsistency.")
                 # This is a failsafe, ideally the cache logic above handles this.
                 temp_output = MODEL.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
                 colbert_vectors_for_scoring = temp_output['colbert_vecs']


            if colbert_vectors_for_scoring and len(colbert_vectors_for_scoring) == len(texts):
                first_text_colbert_vec = colbert_vectors_for_scoring[0]
                scores_relative_to_first = []
                for i in range(1, len(colbert_vectors_for_scoring)):
                    candidate_colbert_vec = colbert_vectors_for_scoring[i]
                    score = MODEL.colbert_score(first_text_colbert_vec, candidate_colbert_vec)
                    if hasattr(score, 'item'): score = score.item()
                    scores_relative_to_first.append({
                        "query_text_index": 0,
                        "candidate_text_index": i,
                        "score": score
                    })
                result['colbert_scores_relative_to_first'] = scores_relative_to_first
            else:
                logger.warning("Could not compute colbert_pairwise_scores due to missing vectors.")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing /encode request: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error during encoding"}), 500


@app.route('/colbert-similarity', methods=['POST'])
@require_api_key
@wait_for_model
def colbert_similarity_to_query():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Invalid JSON payload"}), 400
        
        query_text = data.get('query_text')
        candidate_texts = data.get('candidate_texts')

        if not query_text or not isinstance(query_text, str):
            return jsonify({"error": "Missing or invalid 'query_text'"}), 400
        if not candidate_texts or not isinstance(candidate_texts, list) or \
           not all(isinstance(t, str) for t in candidate_texts):
            return jsonify({"error": "Missing or invalid 'candidate_texts'"}), 400
        if not candidate_texts: return jsonify({"scores": []}), 200

        all_texts_for_scoring = [query_text] + candidate_texts
        final_colbert_vectors_for_scoring = [None] * len(all_texts_for_scoring)
        texts_to_encode_freshly_map = {} # {original_idx: text_val}
        
        for original_idx, text_val in enumerate(all_texts_for_scoring):
            if text_val in COLBERT_VEC_CACHE:
                logger.debug(f"Cache hit for text '{text_val[:30]}...' in /colbert-similarity")
                final_colbert_vectors_for_scoring[original_idx] = COLBERT_VEC_CACHE[text_val]
            else:
                logger.debug(f"Cache miss for text '{text_val[:30]}...' in /colbert-similarity")
                texts_to_encode_freshly_map[original_idx] = text_val
        
        if texts_to_encode_freshly_map:
            texts_to_encode_values = list(texts_to_encode_freshly_map.values())
            original_indices_of_fresh_texts = list(texts_to_encode_freshly_map.keys())
            
            logger.info(f"Encoding {len(texts_to_encode_values)} texts freshly for /colbert-similarity.")
            fresh_output = MODEL.encode(
                texts_to_encode_values, 
                return_dense=False, return_sparse=False, return_colbert_vecs=True
            )
            fresh_vecs = fresh_output.get('colbert_vecs')
            
            if fresh_vecs and len(fresh_vecs) == len(texts_to_encode_values):
                for i, vec in enumerate(fresh_vecs):
                    original_idx = original_indices_of_fresh_texts[i]
                    text_val = texts_to_encode_values[i]
                    final_colbert_vectors_for_scoring[original_idx] = vec
                    COLBERT_VEC_CACHE[text_val] = vec 
                    logger.debug(f"Cached fresh Colbert vector for '{text_val[:30]}...'")
            else:
                logger.error("Failed to get valid Colbert vectors for subset in /colbert-similarity.")
                return jsonify({"error": "Failed to compute some Colbert vectors"}), 500
        else:
            logger.info("All Colbert vectors for /colbert-similarity found in cache.")

        if any(v is None for v in final_colbert_vectors_for_scoring):
            logger.error("Internal error: Some Colbert vectors are None after processing in /colbert-similarity.")
            return jsonify({"error": "Internal error preparing Colbert vectors for scoring"}), 500
            
        query_colbert_vec = final_colbert_vectors_for_scoring[0]
        candidate_colbert_vecs_for_scoring = final_colbert_vectors_for_scoring[1:]
        
        scores = []
        for cand_vec in candidate_colbert_vecs_for_scoring:
            score = MODEL.colbert_score(query_colbert_vec, cand_vec)
            if hasattr(score, 'item'): score = score.item()
            scores.append(score)
            
        return jsonify({"scores": scores}), 200

    except Exception as e:
        logger.error(f"Error in /colbert-similarity: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred during colbert similarity"}), 500

# --- Main ---
if __name__ == '__main__':
    # Start BGE M3 model initialization in a background thread
    # This is critical for services that need the model ready before serving requests
    # (e.g. Cloud Run, which might send requests soon after container start)
    start_model_initialization_thread()
    
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # For production (like Cloud Run), Gunicorn is recommended.
    # The Dockerfile CMD will use Gunicorn.
    # This app.run() is mainly for local development.
    logger.info(f"Starting Flask development server on port {port} with debug_mode={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 