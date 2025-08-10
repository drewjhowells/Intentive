# analyze_emotion_distribution.py

import os
import pandas as pd
from collections import Counter
import re

# --- CONFIGURE THIS ---
DATA_DIR = r"C:\Repos\Intentive\ai_structure\data\emotional_analyzer_data"

# Label map for .txt datasets
label_map = {
    0: 'joy', 1: 'fear', 2: 'anger', 3: 'sadness',
    4: 'disgust', 5: 'shame', 6: 'guilt'
}

# --- Helpers ---
def parse_labeled_text_file(filepath, label_map):
    """Parses custom-labeled .txt files with [0. 1. 0. ...] format"""
    data = []
    label_pattern = re.compile(r'^\[\s*((?:\d\.\s*)+)\]')
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            match = label_pattern.match(line)
            if match:
                values = [float(v) for v in match.group(1).strip().split()]
                labels = [label_map[i] for i, val in enumerate(values) if val == 1.0]
                text = line[match.end():].strip()
                if labels and text:
                    data.append({'labels_list': labels, 'input_text': text})
    return pd.DataFrame(data)

def load_combined_dataframe(data_dir):
    """Loads and combines all CSV and TXT datasets into one DataFrame"""
    datasets = []
    training_files = sorted(os.listdir(data_dir))

    for filename in training_files:
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            if 'labels' in df.columns:
                if filename.startswith("emotions"):
                    df['labels'] = df['labels'].map({0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear'})
                df['labels_list'] = df['labels'].apply(lambda x: [x])
                datasets.append(df[['input_text', 'labels_list']])

        elif filename.endswith(".txt"):
            df = parse_labeled_text_file(filepath, label_map)
            datasets.append(df)

    if not datasets:
        raise ValueError("No valid dataset files found in the directory.")

    return pd.concat(datasets, ignore_index=True)

def count_labels(df):
    """Counts all emotion label occurrences across all samples"""
    df['labels_list'] = df['labels_list'].apply(lambda labels: [str(label) for label in labels])
    counts = Counter(label for sublist in df['labels_list'] for label in sublist)
    return pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).sort_values(by='Count', ascending=False)

# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
    else:
        df = load_combined_dataframe(DATA_DIR)
        label_table = count_labels(df)

        print("\n📊 Emotion Label Frequency Table:\n")
        print(label_table.to_string())
