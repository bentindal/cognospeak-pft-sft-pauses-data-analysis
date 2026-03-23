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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import dataloader
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

N_LIMIT = 20
RANDOM_SEED = 42
K_FOLDS = 5
MIN_TOKENS = 2

loader = dataloader.TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

frames_dict = loader.load_all_transcripts(transcript_paths)
metadata = loader.load_metadata()

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


def build_corpus(dataframes, min_tokens=MIN_TOKENS):
    corpus = []
    skipped = 0
    for df in dataframes:
        tokens = [
            t.strip() 
            for t in df['token'].astype(str).tolist() 
            if t.strip()
        ]

        if len(tokens) < min_tokens:
            skipped += 1
            print("SKIPPING:", tokens)
            continue

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

def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=stemmed_tokenizer,
            token_pattern=None,
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('clf', LinearSVC())
    ])

def build_dataset(task_diagnosis_frames, tasks, diagnoses, min_tokens=2):
    X, y, groups = [], [], []

    skipped = 0 
    for task in tasks:
        for diagnosis in diagnoses:
            frames = task_diagnosis_frames[task][diagnosis]

            for stem, df in frames.items():
                tokens = [
                    t.strip() 
                    for t in df['token'].astype(str).tolist() 
                    if t.strip()
                ]

                if len(tokens) < min_tokens:
                    skipped += 1
                    print("SKIPPING:", tokens)
                    continue

                text = ' '.join(tokens)

                X.append(text)
                y.append(diagnosis)

    print(f"Skipped {skipped} short transcripts")
    return X, y, groups

def run_cv_experiment_with_cm(X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    pipeline = build_pipeline()

    metrics = []
    conf_matrices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        cm = confusion_matrix(y_test, y_pred, labels=TARGET_DIAGNOSES)

        metrics.append((acc, precision, recall, f1))
        conf_matrices.append(cm)

    return np.array(metrics), conf_matrices

def run_task_experiment(task_name, task_diagnosis_frames):
    print(f"\n===== TASK: {task_name} =====")

    X, y = [], []

    for diagnosis in TARGET_DIAGNOSES:
        frames = task_diagnosis_frames[task_name][diagnosis]

        for df in frames.values():
            text = ' '.join(df['token'].astype(str).tolist())
            X.append(text)
            y.append(diagnosis)

    metrics, conf_matrices = run_cv_experiment_with_cm(X, y)

    summarise_results(metrics)
    plot_avg_confusion_matrix(conf_matrices, TARGET_DIAGNOSES, task_name)


def plot_avg_confusion_matrix(conf_matrices, labels, task_name):
    avg_cm = np.mean(conf_matrices, axis=0)

    plt.figure()
    plt.imshow(avg_cm)
    plt.title(f"Average Confusion Matrix ({task_name})")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{avg_cm[i, j]:.2f}",
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def summarise_results(metrics):
    mean = metrics.mean(axis=0)
    std = metrics.std(axis=0)

    print("\n=== Final Results ===")
    print(f"Accuracy: {mean[0]:.3f} ± {std[0]:.3f}")
    print(f"Precision: {mean[1]:.3f} ± {std[1]:.3f}")
    print(f"Recall: {mean[2]:.3f} ± {std[2]:.3f}")
    print(f"F1-score: {mean[3]:.3f} ± {std[3]:.3f}")

def main():
    for task in TARGET_TASKS:
        print(f"Generating visual analysis for {task}...")
        run_task_experiment(task, task_diagnosis_frames)
        plot_tfidf_comparison(task, task_diagnosis_frames[task], N_LIMIT)
    
    X, y, groups = build_dataset(
        task_diagnosis_frames,
        TARGET_TASKS,
        TARGET_DIAGNOSES
    )

    metrics, conf_matrices = run_cv_experiment_with_cm(X, y)

    summarise_results(metrics)

if __name__ == "__main__":
    main()