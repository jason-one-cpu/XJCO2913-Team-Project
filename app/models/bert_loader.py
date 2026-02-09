from transformers import pipeline
import logging

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsBert:
    def __init__(self):
        """
        Initialize the BERT pipeline.
        'ProsusAI/finbert' is trained on financial news, which is inherently
        objective/neutral. This allows us to detect "Neutral" sentiment,
        which is crucial for accurate news bias analysis.
        """
        logger.info("Loading FinBERT model pipeline...")
        try:
            # We explicitly specify the model to ensure 3-class classification
            self.model_name = "ProsusAI/finbert"
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name
            )
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.classifier = None

    def predict(self, text):
        """
        Analyze the sentiment of a SINGLE sentence.

        Args:
            text (str): A single sentence or short text segment.

        Returns:
            dict: {'label': 'POSITIVE'/'NEGATIVE'/'NEUTRAL', 'score': float}
        """
        if not self.classifier:
            return {"error": "Model not ready"}

        if not text or not isinstance(text, str):
            return {"error": "Invalid input"}

        try:
            results = self.classifier(text)

            # FinBERT returns [{'label': 'positive', 'score': 0.9}]
            result = results[0]

            # Map model labels to our project defined labels for frontend consistency
            label_map = {
                "positive": "POSITIVE",
                "negative": "NEGATIVE",
                "neutral": "NEUTRAL"
            }

            original_label = result['label'].lower()
            standardized_label = label_map.get(original_label, "UNKNOWN")

            return {
                "label": standardized_label,
                "score": round(result['score'], 4)
            }

        except Exception as e:
            logger.error(f"Prediction error for text '{text[:20]}...': {e}")
            return {"error": str(e)}


# --- Local Test Block ---
# This block is only used for local testing of the model,
# and will not be invoked during the actual software runtime
if __name__ == "__main__":
    print("--- Testing BERT Model ---")
    bert = NewsBert()

    test_sentences = [
        "The company reported a record-breaking profit increase of 20%.",  # Should be POSITIVE
        "The stock market crashed due to the unexpected policy change.",  # Should be NEGATIVE
        "The meeting is scheduled for next Tuesday at 10 AM.",  # Should be NEUTRAL
        "One year ago today, President Donald J. Trump returned to office with a resounding mandate to restore prosperity, secure the border, rebuild American strength, and put the American people first. In just 365 days, President Trump has delivered truly transformative results with the most accomplished first year of any presidential term in modern history."
    ]

    for s in test_sentences:
        res = bert.predict(s)
        print(f"\nSentence: {s}")
        print(f"Result: {res}")