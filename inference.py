import torch
from transformers import AutoModel, BertTokenizerFast, MarianMTModel, MarianTokenizer
import torch.nn as nn
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved sentiment model (architecture + weights)
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        output = self.bert(sent_id, attention_mask=mask)
        cls_hs = output.last_hidden_state[:, 0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load tokenizer and BERT backbone
tokenizer = BertTokenizerFast.from_pretrained("sentiment_tokenizer")
bert_base = AutoModel.from_pretrained("bert-base-uncased")

# Load model and weights
sentiment_model = BERT_Arch(bert_base)
sentiment_model.load_state_dict(torch.load("sentiment_model_full_w.pt", map_location=device))
sentiment_model.to(device)
sentiment_model.eval()

# Load translation model/tokenizer
translation_tokenizer = MarianTokenizer.from_pretrained("translation_tokenizer")
translation_model = MarianMTModel.from_pretrained("translation_model").to(device)
translation_model.eval()

# Inference function
def predict_sentiment(texts):
    inputs = tokenizer.batch_encode_plus(texts,
                                         max_length=128,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sentiment_model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, axis=1)
    return preds.cpu().numpy(), probs.cpu().numpy()

def translate_to_hindi(texts):
    inputs = translation_tokenizer(texts, return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=128).to(device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.no_grad():
        translated = translation_model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return translation_tokenizer.batch_decode(translated, skip_special_tokens=True)


'''
# Example input
texts = [
    "This product is amazing! I loved it and will buy again.",
    "Terrible quality, it broke after one use.",
]


# Generate outputs
sentiments = predict_sentiment(texts)
translations = translate_to_hindi(texts)

# Print results
for orig, hi, sent in zip(texts, translations, sentiments):
    sentiment_label = "Positive" if sent == 1 else "Negative"
    print(f"\nText: {orig}")
    print(f"Translation: {hi}")
    print(f"Sentiment: {sentiment_label}")
'''