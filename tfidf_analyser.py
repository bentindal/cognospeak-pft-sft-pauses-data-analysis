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

loader = dataloader.TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

frames_dict = loader.load_all_transcripts(transcript_paths)
metadata = loader.load_metadata()

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

for task in TARGET_TASKS:
    total_task = sum(len(task_diagnosis_frames[task][diag]) for diag in TARGET_DIAGNOSES)
    print(f"Loaded {total_task} {task} transcripts")
    for diagnosis in TARGET_DIAGNOSES:
        print(f"  {diagnosis}: {len(task_diagnosis_frames[task][diagnosis])}")

def build_corpus(dataframes):
    corpus = []
    for df in dataframes:
        # Join all tokens in this DataFrame into a single string
        text = ' '.join(df['token'].astype(str).tolist())
        corpus.append(text)
    return corpus

N_LIMIT = 20

for task in TARGET_TASKS:
    print("\n" + "=" * 50)
    print(f"{task} Analysis by Diagnosis")
    print("=" * 50)

    for diagnosis in TARGET_DIAGNOSES:
        diagnosis_frames = task_diagnosis_frames[task][diagnosis]
        corpus = build_corpus(diagnosis_frames.values())

        print(f"\n[{task} | {diagnosis}] documents: {len(corpus)}")
        if not corpus:
            print("  No documents available")
            continue

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_stats = mean_variance_axis(X, axis=0)
        tfidf_scores = tfidf_stats[0]
        sorted_indices = tfidf_scores.argsort()[::-1]

        print(f"  Top {N_LIMIT} terms:")
        for i in sorted_indices[:N_LIMIT]:
            print(f"    {feature_names[i]}: {tfidf_scores[i]:.4f}")
