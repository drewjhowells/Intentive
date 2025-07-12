"""
Author: Tristan Allen
Purpose: Train model on user input. Take user input and return the emotional content of the input
"""

import re
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
from input_classifier_v1 import text_to_bert_tokens
import torch
from torch import sigmoid
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cpu")



# --- Dataset wrapper ---
class EmotionalIntentDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Core classifier ---
class EmotionalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.output(x)

    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, input_dim: int, hidden_dim: int, output_dim: int, device: str = 'cpu'):
        model = cls(input_dim, hidden_dim, output_dim)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def generate_embeddings(texts, tokenizer, bert_model, batch_size=32):
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []

    for batch_texts in tqdm(loader, desc="Generating BERT Embeddings"):
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()



# --- High-level analyzer ---
class EmotionAnalyzer:
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlb = MultiLabelBinarizer()
        self.model = None

    def train(self, texts: list[str], label_lists: list[list[str]], epochs: int = 10000, batch_size: int = 32, lr: float = 0.001):
        Y = self.mlb.fit_transform(label_lists)
        classes = list(self.mlb.classes_)
        print(f"Detected emotion classes: {classes}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
        X = generate_embeddings(texts, tokenizer, bert_model)


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
        train_ds = EmotionalIntentDataset(X_train, Y_train)
        test_ds = EmotionalIntentDataset(X_test, Y_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        self.model = EmotionalClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=len(classes)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            for batch_X, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
            self.evaluate(test_loader, label_names=classes)
        return X_test, Y_test

    def predict(self, texts: list[str], threshold: float = 0.1) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

        embeddings = text_to_bert_tokens(texts, tokenizer, bert_model, device=device)
        X = generate_embeddings(texts, tokenizer, bert_model)
        
        with torch.no_grad():
            probs = self.model(X).cpu().numpy()
        
        results = []
        for row in probs:
            selected_emotions = [cls for cls, p in zip(self.mlb.classes_, row) if p >= threshold]
            results.append(", ".join(selected_emotions) if selected_emotions else "neutral")
        
        return results

    def save(self, model_path: str, classes_path: str):
        torch.save({'state_dict': self.model.state_dict(), 'classes': list(self.mlb.classes_)}, model_path)
        with open(classes_path, 'w') as f:
            json.dump(list(self.mlb.classes_), f)
        print(f"Model saved to {model_path} and classes to {classes_path}")

    def evaluate(self, data_loader, label_names=None):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in data_loader:
                preds = sigmoid(self.model(X)).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())
        pred_binarized = (np.array(all_preds) >= 0.1).astype(int)
        print(classification_report(all_targets, pred_binarized, target_names=label_names))


    @classmethod
    def load(cls, model_path: str, classes_path: str, input_dim: int = 768, hidden_dim: int = 256, device: str = 'cpu'):
        data = torch.load(model_path, map_location=device)
        with open(classes_path, 'r') as f:
            classes = json.load(f)

        analyzer = cls(input_dim=input_dim, hidden_dim=hidden_dim)
        analyzer.mlb = MultiLabelBinarizer(classes=classes)
        analyzer.mlb.fit([[]])
        analyzer.model = EmotionalClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(classes))
        analyzer.model.load_state_dict(data['state_dict'])
        analyzer.model.eval()
        print(f"Loaded model with classes: {classes}")
        return analyzer

def parse_labeled_text_file(filepath, label_map):
    data = []
    label_pattern = re.compile(r'^\[\s*((?:\d\.\s*)+)\]')

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            match = label_pattern.match(line)
            if match:
                label_vector_str = match.group(1).strip()
                values = [float(v) for v in label_vector_str.split()]
                labels = [label_map[i] for i, val in enumerate(values) if val == 1.0]
                text = line[match.end():].strip()
                if labels and text:
                    data.append({'labels_list': labels, 'input_text': text})
    return pd.DataFrame(data)

# --- Script entry point ---
def main():
    print("Downloading training file 1 . . .")
    training_filepath1 = "C:/Users/trist/OneDrive/Intentive/models/data/emotions(1).csv"
    df1 = pd.read_csv(training_filepath1)
    df1['labels'] = df1['labels'].map({0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear'})
    df1['labels_list'] = df1['labels'].apply(lambda x: [x])
    print("Loaded training file 1.")

    print("Downloading training file 2 . . .")
    training_filepath2 = "C:/Users/trist/OneDrive/Intentive/models/data/tweet_emotions(2).csv"
    df2 = pd.read_csv(training_filepath2)
    df2['labels_list'] = df2['labels'].apply(lambda x: [x])
    print("Loaded training file 2.")

    print("Downloading training file 3 . . .")
    training_filepath3 = "C:/Users/trist/OneDrive/Intentive/models/data/text(3).txt"
    label_map = {0: 'joy', 1: 'fear', 2: 'anger', 3: 'sadness', 4: 'disgust', 5: 'shame', 6: 'guilt'}
    df3 = parse_labeled_text_file(training_filepath3, label_map)
    print("Loaded training file 3.")

    print("Downloading training file 4 . . .")
    training_filepath4 = "C:/Users/trist/OneDrive/Intentive/models/data/Combined_Data(4).csv"
    df4 = pd.read_csv(training_filepath4)
    df4['labels_list'] = df4['labels'].apply(lambda x: [x])
    print("Loaded training file 4.")

    print("Downloading training file 5 . . .")
    training_filepath5 = "C:/Users/trist/OneDrive/Intentive/models/data/combined_emotion(5).csv"
    df5 = pd.read_csv(training_filepath5)
    df5['labels_list'] = df5['labels'].apply(lambda x: [x])
    print("Loaded training file 5.")

    print("Downloading training file 6 . . .")
    training_filepath6 = "C:/Users/trist/OneDrive/Intentive/models/data/combined_sentiment_data(6).csv"
    df6 = pd.read_csv(training_filepath6)
    df6['labels_list'] = df6['labels'].apply(lambda x: [x])
    print("Loaded training file 6.")

    print("Downloading training file 7 . . .")
    training_filepath7 = "C:/Users/trist/OneDrive/Intentive/models/data/emotion_dataset_raw(7).csv"
    df7 = pd.read_csv(training_filepath7)
    df7['labels_list'] = df7['labels'].apply(lambda x: [x])
    print("Loaded training file 7.")

    print("df1 columns:", df1.columns)
    print("df2 columns:", df2.columns)
    print("df3 columns:", df3.columns)
    print("df4 columns:", df4.columns)
    print("df5 columns:", df5.columns)
    print("df6 columns:", df6.columns)
    print("df7 columns:", df7.columns)


    print("Combining Files . . .")
    df = pd.concat([df1[['input_text', 'labels_list']], df2[['input_text', 'labels_list']], df3[['input_text', 'labels_list']], df4[['input_text', 'labels_list']], df5[['input_text', 'labels_list']], df6[['input_text', 'labels_list']], df7[['input_text', 'labels_list']]], ignore_index=True)
    df['labels_list'] = df['labels_list'].apply(lambda labels: [str(label) for label in labels])

    df = df.sample(n=1000, random_state=42)

    print("Files combined.")

    print("Loading Model . . .")
    analyzer = EmotionAnalyzer()
    print("Training Model  . . .")
    X_test, Y_test = analyzer.train(df['input_text'].astype(str).tolist(), df['labels_list'].tolist(), epochs=1000)
    print("Saving Model . . .")
    analyzer.save('emotion_model.pth', 'emotion_classes.json')
    print("Model Saved.")
    print("Evaluating Model . . .")
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_test, dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(EmotionalIntentDataset(X_tensor, Y_tensor), batch_size=32)

    analyzer.evaluate(test_loader)
    print("Model Evaluated.")

if __name__ == "__main__":
    main()
