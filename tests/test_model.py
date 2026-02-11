import unittest
from app.models.bert_loader import NewsBert


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests."""
        print("\n[Test-Info] Loading Model for Testing... this may take a moment.")
        cls.model = NewsBert()

    def test_model_loading(self):
        """Check if the model pipeline is initialized."""
        self.assertIsNotNone(self.model.classifier, "Model classifier should not be None")

    def test_prediction_structure(self):
        """Check if predict returns the correct dictionary structure."""
        text = "This is a neutral statement."
        result = self.model.predict(text)

        self.assertIn('label', result)
        self.assertIn('score', result)

        # FinBERT labels should be standardized to uppercase (POSITIVE, NEGATIVE, NEUTRAL)
        self.assertIn(result['label'], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
        self.assertIsInstance(result['score'], float)

    def test_positive_sentiment(self):
        """Sanity check: Does it recognize obvious positive text?"""
        text = "The company profits doubled, marking a fantastic year."
        result = self.model.predict(text)
        self.assertEqual(result['label'], 'POSITIVE')

    def test_negative_sentiment(self):
        """Sanity check: Does it recognize obvious negative text?"""
        text = "The crisis caused a tragic loss of revenue."
        result = self.model.predict(text)
        self.assertEqual(result['label'], 'NEGATIVE')


if __name__ == '__main__':
    unittest.main()