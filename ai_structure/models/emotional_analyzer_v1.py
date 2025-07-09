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
import torch
from torch import sigmoid
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

device = torch.device("cpu")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def text_to_bert_tokens(texts, tokenizer, bert_model, batch_size=32, device="cpu"):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Encoding batch {i // batch_size + 1} / {len(texts) // batch_size + 1}")  # Add this
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


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

def generate_embeddings(texts, tokenizer, bert_model, batch_size=16, save_path=None):
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []

    num_samples = len(texts)
    total_batches = (num_samples + batch_size - 1) // batch_size

    existing_batches = sorted(
        [f for f in os.listdir(os.path.dirname(save_path)) if f.startswith(os.path.basename(save_path) + "_batch_")],
        key=lambda f: int(f.split('_batch_')[-1].split('.')[0])
    )
    completed_batches = len(existing_batches)

    if completed_batches >= total_batches:
        print(f"All BERT embeddings already saved ({completed_batches} batches). Skipping generation.")
        return

    print(f"Resuming BERT embedding generation from batch {completed_batches} of {total_batches} total batches.")
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []

    for batch_idx, batch_texts in enumerate(tqdm(loader, desc="Generating BERT Embeddings")):
        if batch_idx < completed_batches:
            continue  # Skip already saved batches

        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        if save_path:
            batch_save_path = f"{save_path}_batch_{batch_idx}.npy"
            np.save(batch_save_path, cls_embeddings.cpu().numpy())
        else:
            all_embeddings.append(cls_embeddings.cpu())

    if not save_path:
        return torch.cat(all_embeddings, dim=0).numpy()



# --- High-level analyzer ---
class EmotionAnalyzer:
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlb = MultiLabelBinarizer()
        self.model = None

    def train(self, texts: list[str], label_lists: list[list[str]], epochs: int = 10000, start_epoch=0, batch_size: int = 8, lr: float = 0.001):
        checkpoints_dir = os.path.join(BASE_DIR, "models", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        Y = self.mlb.fit_transform(label_lists)
        classes = list(self.mlb.classes_)
        print(f"Detected emotion classes: {classes}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

        embedding_dir = os.path.join(BASE_DIR, "models", "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        embedding_path = os.path.join(embedding_dir, "bert_embeddings")

        existing_batches = [f for f in os.listdir(embedding_dir) if f.startswith("bert_embeddings_batch_")]
        if not existing_batches:
            print("Generating and saving BERT embeddings...")
            generate_embeddings(texts, tokenizer, bert_model, save_path=embedding_path)
        else:
            sorted_batches = sorted(existing_batches, key=lambda f: int(f.split('_batch_')[-1].split('.')[0]))
            print(f"Reusing saved BERT embeddings from disk.")
            print(f"Found {len(sorted_batches)} saved embedding batches. Starting from batch 0, using these files:")
            for batch_file in sorted_batches:
                print(f" - {batch_file}")

        all_embeddings = []
        embedding_files = sorted(
            [f for f in os.listdir(embedding_dir) if f.startswith("bert_embeddings_batch_")],
            key=lambda f: int(f.split('_batch_')[-1].split('.')[0])
        )

        for file in embedding_files:
            emb = np.load(os.path.join(embedding_dir, file))
            all_embeddings.append(emb)
        X = np.concatenate(all_embeddings, axis=0)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
        train_ds = EmotionalIntentDataset(X_train, Y_train)
        test_ds = EmotionalIntentDataset(X_test, Y_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        self.model = EmotionalClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=len(classes)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(start_epoch, epochs):
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

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoints_dir, f"emotion_checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'classes': classes,
                }, checkpoint_path)

        final_checkpoint_path = os.path.join(checkpoints_dir, "emotion_checkpoint_final.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': classes,
        }, final_checkpoint_path)
        print(f"Final checkpoint saved at {final_checkpoint_path}.")


        return X_test, Y_test

    def predict(self, texts: list[str], threshold: float = 0.1) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

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

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = EmotionalClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=len(checkpoint['classes'])
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mlb = MultiLabelBinarizer(classes=checkpoint['classes'])
        self.mlb.fit([[]])
        print(f"Checkpoint loaded from {checkpoint_path} at epoch {checkpoint['epoch']}.")
        return checkpoint['epoch']


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
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(BASE_DIR, "data", "emotional_analyzer_data")

    print("Looking for training files in:", data_dir)

    training_files = sorted(os.listdir(data_dir))
    datasets = []

    for filename in training_files:
        filepath = os.path.join(data_dir, filename)
        print(f"Loading {filename}...")

        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            if 'labels' in df.columns:
                if filename.startswith("emotions"):  # Example special handling
                    df['labels'] = df['labels'].map({0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear'})
                df['labels_list'] = df['labels'].apply(lambda x: [x])
            else:
                raise ValueError(f"No 'labels' column found in {filename}")
            datasets.append(df)

        elif filename.endswith(".txt"):
            # Example label map; adjust per your needs
            label_map = {0: 'joy', 1: 'fear', 2: 'anger', 3: 'sadness', 4: 'disgust', 5: 'shame', 6: 'guilt'}
            df = parse_labeled_text_file(filepath, label_map)
            datasets.append(df)

        else:
            print(f"Skipping unsupported file: {filename}")

    print("All files loaded successfully.")

    df = pd.concat([df[['input_text', 'labels_list']] for df in datasets], ignore_index=True)
    df['labels_list'] = df['labels_list'].apply(lambda labels: [str(label) for label in labels])

    print("Files combined.")

    print("Loading Model . . .")
    analyzer = EmotionAnalyzer()
    start_epoch = 0
    checkpoint_path = "emotion_checkpoint_epoch_10.pth"
    if os.path.exists(checkpoint_path):
        start_epoch = analyzer.load_checkpoint(checkpoint_path)
    print(f"Training Model at epoch {start_epoch}. . .")
    X_test, Y_test = analyzer.train(
    df['input_text'].astype(str).tolist(),
    df['labels_list'].tolist(),
    epochs=1000,
    start_epoch=start_epoch
    )
    print("Saving Model . . .")
    analyzer.save('emotion_model.pth', 'emotion_classes.json')
    print("Model Saved.")
    print("Evaluating Model . . .")
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_test, dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(EmotionalIntentDataset(X_tensor, Y_tensor), batch_size=8)

    analyzer.evaluate(test_loader)
    print("Model Evaluated.")

if __name__ == "__main__":
    main()
