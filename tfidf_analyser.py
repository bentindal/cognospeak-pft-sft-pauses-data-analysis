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

# Separate transcripts by task type (SFT and PFT)
sft_frames = {}
pft_frames = {}

for stem, df in frames_dict.items():
    if 'task' in df.columns:
        task = df['task'].iloc[0] if len(df) > 0 else 'UNKNOWN'
        if task == 'SFT':
            sft_frames[stem] = df
        elif task == 'PFT':
            pft_frames[stem] = df
    else:
        # Fallback: infer from stem if task column not available
        if 'SFT' in stem.upper():
            sft_frames[stem] = df
        elif 'PFT' in stem.upper():
            pft_frames[stem] = df

print(f"Loaded {len(sft_frames)} SFT transcripts")
print(f"Loaded {len(pft_frames)} PFT transcripts")

frames = frames_dict.values()

def build_corpus(dataframes):
    corpus = []
    for df in dataframes:
        # Join all tokens in this DataFrame into a single string
        text = ' '.join(df['token'].astype(str).tolist())
        corpus.append(text)
    return corpus

# Build corpus for SFT transcripts
sft_corpus = build_corpus(sft_frames.values())
print(f"\nSFT corpus built with {len(sft_corpus)} documents")

# Build corpus for PFT transcripts
pft_corpus = build_corpus(pft_frames.values())
print(f"PFT corpus built with {len(pft_corpus)} documents")

# Analyze SFT
print("\n" + "="*50)
print("SFT Analysis")
print("="*50)
if sft_corpus:
    sft_vectorizer = TfidfVectorizer()
    sft_X = sft_vectorizer.fit_transform(sft_corpus)
    sft_feature_names = sft_vectorizer.get_feature_names_out()
    sft_tfidf_scores = np.asarray(sft_X.mean(axis=0)).flatten()
    sft_sorted_indices = sft_tfidf_scores.argsort()[::-1]
    
    n_limit = 20
    print("Top 20 terms:")
    for i in sft_sorted_indices[:n_limit]:
        print(f"  {sft_feature_names[i]}: {sft_tfidf_scores[i]:.4f}")

# Analyze PFT
print("\n" + "="*50)
print("PFT Analysis")
print("="*50)
if pft_corpus:
    pft_vectorizer = TfidfVectorizer()
    pft_X = pft_vectorizer.fit_transform(pft_corpus)
    pft_feature_names = pft_vectorizer.get_feature_names_out()
    pft_tfidf_scores = np.asarray(pft_X.mean(axis=0)).flatten()
    pft_sorted_indices = pft_tfidf_scores.argsort()[::-1]
    
    n_limit = 20
    print("Top 20 terms:")
    for i in pft_sorted_indices[:n_limit]:
        print(f"  {pft_feature_names[i]}: {pft_tfidf_scores[i]:.4f}")
