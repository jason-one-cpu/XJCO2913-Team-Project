from transformers import pipeline
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsBert:
    def __init__(self):
        """
        Initialize the BERT pipeline for sentiment analysis.
        We use a distilled version of BERT which is faster and lighter,
        perfect for a web application MVP.
        """
        logger.info("Loading BERT model pipeline...")
        try:
            # specifically specifying the model ensures consistency
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("BERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.classifier = None

    def predict(self, text):
        """
        Analyze the sentiment of the input text.

        Args:
            text (str): The news article text to analyze.

        Returns:
            dict: A dictionary containing 'label' (POSITIVE/NEGATIVE) and 'score'.
                  Returns None if model is not loaded or input is invalid.
        """
        if not self.classifier:
            logger.error("Model not initialized.")
            return {"error": "Model not ready"}

        if not text or not isinstance(text, str):
            logger.warning("Invalid input text.")
            return {"error": "Invalid input"}

        try:
            # BERT models have a hard limit of 512 tokens (roughly 2000-2500 characters).
            # Tell the pipeline to truncate inputs that exceed the maximum length.
            results = self.classifier(text, truncation=True, max_length=512)

            # The pipeline returns a list of dicts: [{'label': 'POSITIVE', 'score': 0.99}]
            result = results[0]

            return {
                "label": result['label'],  # POSITIVE or NEGATIVE
                "score": round(result['score'], 4)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}


# Simple test block to verify it works when running this file directly
if __name__ == "__main__":
    print("Testing NewsBert class...")
    bert = NewsBert()

    test_cases = [
        "The economy is booming and citizens are happier than ever.",
        "The catastrophic failure of the policy led to misery.",
        "The meeting is scheduled for 5 PM.",
        "The meeting is canceled."
    ]

    for t in test_cases:
        print(f"\nText: {t}")
        print(f"Result: {bert.predict(t)}")