#!/usr/bin/env python3
"""
TF-IDF Analysis for Transcript Words

Analyzes vocabulary patterns across transcripts using TF-IDF (Term Frequency-Inverse Document Frequency).
This helps identify:
- Words that are distinctive to specific recordings
- Common vs. rare words in the dataset
- Word importance for classification tasks
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.sparsefuncs import mean_variance_axis
import dataloader
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

loader = dataloader.TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

frames_dict = loader.load_all_transcripts(transcript_paths)
metadata = loader.load_metadata()

N_LIMIT = 20
stemmer = PorterStemmer()

def stemmed_tokenizer(text):
    # Lowercase and spip install --upgrade piptem each word
    tokens = re.split(r'\W+', text.lower())
    return [stemmer.stem(w) for w in tokens if len(w) > 1]


# Map each recording ID to diagnosis class (DEMENTIA, MCI, HC)
diagnosis_by_recording = {
    row['recording']: row['diagnosis']
    for _, row in metadata[['recording', 'diagnosis']].dropna().iterrows()
}

# Separate transcripts by task and diagnosis class
TARGET_TASKS = ('SFT', 'PFT')
TARGET_DIAGNOSES = ('DEMENTIA', 'MCI', 'HC')

task_diagnosis_frames = {
    task: {diagnosis: {} for diagnosis in TARGET_DIAGNOSES}
    for task in TARGET_TASKS
}

for stem, df in frames_dict.items():
    if len(df) == 0:
        continue

    if 'task' in df.columns:
        task = str(df['task'].iloc[0]).upper()
    else:
        # Fallback: infer from stem if task column not available
        task = 'SFT' if 'SFT' in stem.upper() else 'PFT' if 'PFT' in stem.upper() else 'UNKNOWN'

    recording = str(df['recording'].iloc[0]) if 'recording' in df.columns else None
    diagnosis = diagnosis_by_recording.get(recording, 'UNKNOWN')

    if task in TARGET_TASKS and diagnosis in TARGET_DIAGNOSES:
        task_diagnosis_frames[task][diagnosis][stem] = df
        
def build_corpus(dataframes):
    corpus = []
    for df in dataframes:
        # Join all tokens in this DataFrame into a single string
        text = ' '.join(df['token'].astype(str).tolist())
        corpus.append(text)
    return corpus

def plot_tfidf_comparison(task_name, task_data, n_limit=N_LIMIT):
    """
    Creates a side-by-side bar chart for HC, MCI, and DEMENTIA.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
    fig.suptitle(f"Top {n_limit} Stemmed Terms: {task_name} Task", fontsize=18, fontweight='bold')
    
    colors = {'HC': '#2ecc71', 'MCI': '#f1c40f', 'DEMENTIA': '#e74c3c'}

    for idx, diagnosis in enumerate(['HC', 'MCI', 'DEMENTIA']):
        ax = axes[idx]
        diagnosis_frames = task_data.get(diagnosis, {})
        corpus = build_corpus(diagnosis_frames.values())

        if not corpus:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.set_title(f"{diagnosis}")
            continue

        vectorizer = TfidfVectorizer(
            tokenizer=stemmed_tokenizer,
            token_pattern=None
        )
        
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF across all documents in this group
        tfidf_stats = mean_variance_axis(X, axis=0)
        mean_scores = tfidf_stats[0]
        
        # Sort and get top N
        sorted_indices = mean_scores.argsort()[::-1][:n_limit]
        top_terms = [feature_names[i] for i in sorted_indices]
        top_values = [mean_scores[i] for i in sorted_indices]

        # Plotting
        ax.barh(top_terms, top_values, color=colors.get(diagnosis, 'skyblue'))
        ax.set_title(f"{diagnosis} (n={len(corpus)})", fontsize=14)
        ax.invert_yaxis()  # Highest scores at the top
        ax.set_xlabel("Mean TF-IDF Score")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

for task in TARGET_TASKS:
    print(f"Generating visual analysis for {task}...")
    plot_tfidf_comparison(task, task_diagnosis_frames[task], N_LIMIT)