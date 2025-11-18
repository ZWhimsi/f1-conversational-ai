"""
Flask Backend API for F1 Demo
Serves the model API and frontend.
"""

from flask import Flask, render_template, request, jsonify
import logging
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Initialize handler (local model or remote API)
if config.USE_REMOTE_API:
    from remote_api_handler import RemoteAPIHandler
    model_handler = RemoteAPIHandler(
        api_type=config.REMOTE_API_TYPE,
        api_key=config.REMOTE_API_KEY,
        api_url=config.REMOTE_API_URL,
        model_id=config.REMOTE_MODEL_ID
    )
    logger.info(f"Using {config.REMOTE_API_TYPE} API instead of local model")
    logger.info(f"Model: {config.REMOTE_MODEL_ID}")
else:
    from model_handler import ModelHandler
    model_handler = ModelHandler(
        model_path=config.MODEL_PATH,
        device=config.DEVICE,
        use_quantization=config.USE_QUANTIZATION,
        quantization_bits=config.QUANTIZATION_BITS,
        base_model_path=config.BASE_MODEL_PATH if hasattr(config, 'BASE_MODEL_PATH') else None
    )
    logger.info(f"Using local model: {config.MODEL_PATH}")
    if hasattr(config, 'BASE_MODEL_PATH') and config.BASE_MODEL_PATH:
        logger.info(f"Base model for LoRA: {config.BASE_MODEL_PATH}")

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API requests."""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        response = model_handler.generate_response(
            prompt=user_message,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            system_prompt=config.SYSTEM_PROMPT
        )
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler.loaded,
        'device': config.DEVICE,
        'model_type': 'lora' if hasattr(model_handler, 'is_lora') and model_handler.is_lora else 'full'
    })

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting F1 Demo Application...")
    
    if config.USE_REMOTE_API:
        logger.info("Using remote API - no local model loading needed")
        if model_handler.loaded:
            logger.info("Remote API configured successfully!")
        else:
            logger.warning("Remote API not configured. Please set API key in config.py")
    else:
        logger.info(f"Loading model from: {config.MODEL_PATH}")
        if hasattr(config, 'BASE_MODEL_PATH') and config.BASE_MODEL_PATH:
            logger.info(f"Base model: {config.BASE_MODEL_PATH}")
        if not model_handler.load_model():
            logger.error("Failed to load model. Please check config.py")
            logger.error("Options:")
            logger.error("  1. Use a smaller model")
            logger.error("  2. Enable quantization (already enabled by default)")
            logger.error("  3. Use CPU mode (set USE_GPU=false)")
            logger.error("  4. Use remote API (set USE_REMOTE_API=true)")
            logger.error("  5. For LoRA models, ensure BASE_MODEL_PATH is set correctly")
    
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

Serves the model API and frontend.
"""

from flask import Flask, render_template, request, jsonify
import logging
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Initialize handler (local model or remote API)
if config.USE_REMOTE_API:
    from remote_api_handler import RemoteAPIHandler
    model_handler = RemoteAPIHandler(
        api_type=config.REMOTE_API_TYPE,
        api_key=config.REMOTE_API_KEY,
        api_url=config.REMOTE_API_URL,
        model_id=config.REMOTE_MODEL_ID
    )
    logger.info(f"Using {config.REMOTE_API_TYPE} API instead of local model")
    logger.info(f"Model: {config.REMOTE_MODEL_ID}")
else:
    from model_handler import ModelHandler
    model_handler = ModelHandler(
        model_path=config.MODEL_PATH,
        device=config.DEVICE,
        use_quantization=config.USE_QUANTIZATION,
        quantization_bits=config.QUANTIZATION_BITS,
        base_model_path=config.BASE_MODEL_PATH if hasattr(config, 'BASE_MODEL_PATH') else None
    )
    logger.info(f"Using local model: {config.MODEL_PATH}")
    if hasattr(config, 'BASE_MODEL_PATH') and config.BASE_MODEL_PATH:
        logger.info(f"Base model for LoRA: {config.BASE_MODEL_PATH}")

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API requests."""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        response = model_handler.generate_response(
            prompt=user_message,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            system_prompt=config.SYSTEM_PROMPT
        )
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler.loaded,
        'device': config.DEVICE,
        'model_type': 'lora' if hasattr(model_handler, 'is_lora') and model_handler.is_lora else 'full'
    })

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting F1 Demo Application...")
    
    if config.USE_REMOTE_API:
        logger.info("Using remote API - no local model loading needed")
        if model_handler.loaded:
            logger.info("Remote API configured successfully!")
        else:
            logger.warning("Remote API not configured. Please set API key in config.py")
    else:
        logger.info(f"Loading model from: {config.MODEL_PATH}")
        if hasattr(config, 'BASE_MODEL_PATH') and config.BASE_MODEL_PATH:
            logger.info(f"Base model: {config.BASE_MODEL_PATH}")
        if not model_handler.load_model():
            logger.error("Failed to load model. Please check config.py")
            logger.error("Options:")
            logger.error("  1. Use a smaller model")
            logger.error("  2. Enable quantization (already enabled by default)")
            logger.error("  3. Use CPU mode (set USE_GPU=false)")
            logger.error("  4. Use remote API (set USE_REMOTE_API=true)")
            logger.error("  5. For LoRA models, ensure BASE_MODEL_PATH is set correctly")
    
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
