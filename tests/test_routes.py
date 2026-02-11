import unittest
import json
from app import create_app


class TestAPI(unittest.TestCase):
    def setUp(self):
        """Set up a test client before each test."""
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.testing = True

    def test_homepage(self):
        """Test if the homepage loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Check if our project title is in the HTML
        self.assertIn(b'News Spectrum', response.data)

    def test_predict_empty_json(self):
        """Test /predict endpoint with no JSON data."""
        response = self.client.post('/predict',
                                    data=json.dumps({}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid or empty text', response.data)

    def test_predict_valid_text(self):
        """Test /predict endpoint with valid text."""
        sample_text = "The economy is doing great."
        response = self.client.post('/predict',
                                    data=json.dumps({'text': sample_text}),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Expect a list of results
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)

        # Check structure of the first result
        first_sentence = data[0]
        self.assertIn('label', first_sentence)
        self.assertIn('score', first_sentence)
        self.assertIn('sentence', first_sentence)


if __name__ == '__main__':
    unittest.main()