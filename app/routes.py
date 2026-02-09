from flask import Blueprint, render_template, request, jsonify
from app.models.bert_loader import NewsBert
from sentence_splitter import SentenceSplitter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# --- Global Initialization ---
# Load the BERT Model
try:
    logger.info("Initializing BERT model...")
    bert_model = NewsBert()
except Exception as e:
    logger.error(f"Critical Error: Failed to initialize BERT model: {e}")
    bert_model = None

# Load the Sentence Splitter
# We initialize this once to avoid overhead on every request.
try:
    logger.info("Initializing Sentence Splitter...")
    splitter = SentenceSplitter(language='en')
except Exception as e:
    logger.error(f"Failed to initialize Splitter: {e}")
    splitter = None


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    """
    Sprint 3 Upgrade: Sentence-level Analysis.
    Input: { "text": "Full article text..." }
    Output: [
        {"sentence": "Sentence 1", "label": "POSITIVE", "score": 0.9},
        {"sentence": "Sentence 2", "label": "NEUTRAL", "score": 0.8}
    ]
    """
    # Request Validation
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    full_text = data.get('text', '')

    if not full_text or not isinstance(full_text, str) or not full_text.strip():
        return jsonify({"error": "Invalid or empty text provided"}), 400

    if not bert_model or not splitter:
        return jsonify({"error": "Server services not initialized"}), 500

    try:
        # Split Text into Sentences
        # We break the Block into Pieces
        sentences = splitter.split(full_text)

        results = []

        # Batch Prediction
        # Loop through each sentence and get its individual sentiment.
        for sentence in sentences:
            # Skip empty nonsense
            if not sentence.strip():
                continue

            # Call Webb's model
            prediction = bert_model.predict(sentence)

            # If model returns error, handle it and don't crash the whole loop
            if "error" in prediction:
                logger.warning(f"Model error on sentence: {sentence[:10]}...")
                label = "UNKNOWN"
                score = 0.0
            else:
                label = prediction['label']
                score = prediction['score']

            # Append structured result
            results.append({
                "sentence": sentence,
                "label": label,
                "score": score
            })

        # Return List of Results
        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Prediction route error: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500