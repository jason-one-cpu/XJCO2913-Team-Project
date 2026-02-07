from transformers import BertTokenizer, BertForSequenceClassification
import torch

class NewsBert:
    def __init__(self):
        print("Loading BERT model...")
        # For Sprint 1, we just load the base uncased model to prove it works
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        print("BERT model loaded successfully.")

    def predict(self, text):
        # A dummy prediction function for Sprint 1
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits