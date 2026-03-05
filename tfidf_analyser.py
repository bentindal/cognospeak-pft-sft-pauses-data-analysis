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
import dataloader
import numpy as np

loader = dataloader.TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

frames_dict = loader.load_all_transcripts(transcript_paths)

frames = frames_dict.values()

corpus = []
for df in frames_dict.values():
    # Join all tokens in this DataFrame into a single string
    text = ' '.join(df['token'].astype(str).tolist())
    corpus.append(text)

print(corpus)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = np.asarray(X.mean(axis=0)).flatten()
sorted_indices = tfidf_scores.argsort()[::-1]

n_limit = 20
print("==================Top 20==================")
for i in sorted_indices[:n_limit]:
    print(f"{feature_names[i]}: {tfidf_scores[i]:.4f}")

#bottom 20 
print("==================Bottom 20==================")
for i in sorted_indices[-n_limit:]:
    print(f"{feature_names[i]}: {tfidf_scores[i]:.4f}")
