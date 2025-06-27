"""
Author: Tristan Allen
Purpose: Train model on user input. Take user input and return the classification of the input
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm

bert_token = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # Disables dropout of nerual nodes


def text_to_bert_tokens(text_list, tokenizer, model):
    """
    Purpose: Take a list of text, tokenize it, and runit through a Bert model
    Returns: A 768 size vector for each text in the list
    """
    embeddings = []
    for text in tqdm(text_list):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # get [CLS] token representation
        embeddings.append(cls_embedding.squeeze().numpy())
    return embeddings


class GoalIntentDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GoalClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=14):
        super(GoalClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def main():
    # Collect User input and format as lists not strings
    user_inputs_filepath = "C:/Users/trist/OneDrive/Productivity App/Input Classification/input_classifier_v1_dataset.csv"
    user_inputs = pd.read_csv(user_inputs_filepath)
    user_inputs['labels_list'] = user_inputs['labels'].str.split(';')
    
    # Find the multiple labels present in the data
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(user_inputs['labels_list'])
    print("Classes: ", mlb.classes_)

    # Convert text to Bert Tokens
    texts = user_inputs['input_text'].tolist()
    X = text_to_bert_tokens(texts, bert_token, bert_model)
    print(f"Embeddings: {len(X)}/{len(user_inputs['input_text'])} | Shape: {X[0].shape}")

    # Split the data for training and testing
    X = np.array(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 42)

    # Training the neural network
    train_dataset = GoalIntentDataset(X_train, Y_train)
    test_dataset = GoalIntentDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Define model, loss, optimizer
    model = GoalClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # Evaluate the model
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = outputs.cpu().numpy()
            all_preds.append(probs)
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    binary_preds = (all_preds > 0.5).astype(int)

    print("\nEvaluation Metrics:")
    print(classification_report(
        all_targets,
        binary_preds,
        target_names=mlb.classes_,
        zero_division=0
    ))


if __name__ == "__main__":
    main()