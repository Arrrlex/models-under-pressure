import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from wordcloud import WordCloud

# Load pre-trained BERT tokenizer & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)  # type: ignore
bert_model.eval()

# Load dataset (replace with your dataset)
file_path = "data/inputs/prompts_04_03_25_model-4o.jsonl"
data = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)  # Ensure it has 'prompt' and 'high_stakes' columns
X_train, X_test, y_train, y_test = train_test_split(
    df["prompt"], df["high_stakes"], test_size=0.2, random_state=42
)


# Function to extract BERT embeddings (CLS token)
def extract_bert_embeddings(
    text_list: list[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
) -> tuple[np.ndarray, list[list[str]]]:
    embeddings = []
    tokenized_texts = []
    for text in text_list:
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)  # type: ignore
        attention_mask = encoding["attention_mask"].to(device)  # type: ignore

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Use CLS token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        embeddings.append(cls_embedding)

        # Store tokenized words (for later feature importance analysis)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokenized_texts.append(tokens)

    return np.array(embeddings), tokenized_texts


# Extract embeddings and tokens
X_train_embeddings, train_tokens = extract_bert_embeddings(
    X_train.tolist(), tokenizer, bert_model, device
)
X_test_embeddings, test_tokens = extract_bert_embeddings(
    X_test.tolist(), tokenizer, bert_model, device
)

# Train Logistic Regression classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_embeddings, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_embeddings)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute feature importance
feature_importance = np.abs(clf.coef_[0])

# Map feature importance to tokens
token_importance = {}
for tokens, importance in zip(train_tokens, feature_importance):
    for token in tokens:
        if token not in token_importance:
            token_importance[token] = 0
        token_importance[token] += importance

# Normalize importance scores
max_importance = max(token_importance.values())
token_importance = {k: v / max_importance for k, v in token_importance.items()}
# print(token_importance for specific token)
for token, importance in token_importance.items():
    print(f"{token}: {importance}")
# Generate word cloud from token importance
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(token_importance)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Feature Importance per Token (BERT)")
plt.show()
