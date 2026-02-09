from flask import Blueprint, render_template, request, jsonify
from app.models.bert_loader import NewsBert
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# Model Initialization
# Initialize the model globally here so it loads only ONCE when the app starts.
try:
    logger.info("Initializing BERT model for the application...")
    bert_model = NewsBert()
except Exception as e:
    logger.error(f"Critical Error: Failed to initialize BERT model: {e}")
    bert_model = None


@main.route('/')
def index():
    """
    Renders the homepage.
    """
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint to predict sentiment of a given text.
    Expected JSON input: { "text": "Some news content..." }
    """
    # Input Validation
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    text = data.get('text', '')

    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Invalid or empty text provided"}), 400

    # Check Model Status
    if not bert_model:
        return jsonify({"error": "Model is not initialized on server"}), 500

    # Model Inference
    try:
        # call the predict method from Webb's code
        result = bert_model.predict(text)

        # Check for internal model errors
        if "error" in result:
            return jsonify(result), 500

        # Return Success Response
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction route error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500