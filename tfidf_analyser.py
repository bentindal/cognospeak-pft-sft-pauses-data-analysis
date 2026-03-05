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

loader = dataloader.TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

frames = loader.load_all_transcripts(transcript_paths)

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
print(X.shape)
print(X)
