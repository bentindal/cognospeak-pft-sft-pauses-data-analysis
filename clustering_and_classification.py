import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from dataloader import TranscriptLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import re
from sklearn.decomposition import PCA

def clean_id(raw_id):
    """
    Standardizes IDs by removing prefixes and leading zeros.
    Converts 'Process-rec_097' or 'rec-97' into '97'.
    """
    # Removes Process-, rec-, rec_, and non-numeric prefixes
    clean = re.sub(r'^(Process-|rec[-_])', '', str(raw_id), flags=re.IGNORECASE)
    # Remove leading zeros (e.g 097 -> 97)
    return clean.lstrip('0')

def generate_diagnosis_embeddings(data_dir='data'):
    loader = TranscriptLoader(data_dir=Path(data_dir))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    metadata = pd.read_csv('data/speechpause_mata.csv')
    metadata['anyon_IDs'] = metadata['anyon_IDs'].astype(str).str.strip()
    
    transcript_paths = list(Path(data_dir).rglob('*.txt'))
    all_data = []

    for path in transcript_paths:
        # The participant ID is the name of the folder
        folder_id = path.parent.name.strip()
        
        # Match ID in CSV
        merged_row = metadata[metadata['anyon_IDs'] == folder_id]
        if merged_row.empty: continue
            
        # Standardise Diagnosis
        raw_diag = str(merged_row['diagnosis'].iloc[0]).strip().upper()
        if 'DEMENTIA' in raw_diag: diagnosis = 'Dementia'
        elif 'MCI' in raw_diag: diagnosis = 'MCI'
        elif 'HC' in raw_diag: diagnosis = 'HC'
        else: continue

        df = loader.load_transcript(path)
        tokens = df['token'].dropna().astype(str).str.lower().tolist()
        if not tokens: continue
            
        embeddings = model.encode(tokens)
        task_type = loader.infer_task_from_filename(path.name)
        
        for i in range(len(tokens)):
            all_data.append({
                'recording': folder_id,
                'task': task_type,
                'start_time': df['start_time'].iloc[i],
                'end_time': df['end_time'].iloc[i],
                'diagnosis': diagnosis, 
                'embedding': embeddings[i]
            })

    return pd.DataFrame(all_data)

def visualise_by_diagnosis(df, task_filter='SFT'):
    """
    Visualises the semantic space. 
    Shows word labels ONLY if a single recording is being analyzed.
    """
    subset = df[df['task'] == task_filter].copy()
    
    # Bold, high-contrast palette
    color_map = {
        'Dementia': '#e31a1c', # Bright Red
        'MCI': '#ff7f00',      # Bright Orange
        'HC': '#33a02c'        # Bright Green
    }
    
    if subset.empty:
        print(f"No data points found for {task_filter}. Check CSV labels.")
        return
        
    embeddings = np.stack(subset['embedding'].values)
    pca = PCA(n_components=2, random_state=42)
    reduced_vecs = pca.fit_transform(embeddings)
    subset['pc1'] = reduced_vecs[:, 0]
    subset['pc2'] = reduced_vecs[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plots the dots
    for label, color in color_map.items():
        mask = subset['diagnosis'] == label
        if not mask.any(): 
            continue
        
        ax.scatter(
            subset.loc[mask, 'pc1'], 
            subset.loc[mask, 'pc2'], 
            c=color, 
            label=label, 
            alpha=0.9,
            s=80,
            edgecolors='black',
            linewidth=0.8
        )
    
    # Adds labels ONLY if it's a single file
    # We check the number of unique recordings in the original df
    if df['recording'].nunique() == 1:
        for i, row in subset.iterrows():
            ax.annotate(
                row['token'], 
                (row['pc1'], row['pc2']),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center',
                fontsize=9,
                fontweight='bold'
            )
        title_prefix = f"Single Case Analysis ({subset['recording'].iloc[0]})"
    else:
        title_prefix = "Dataset-wide Semantic Map"

    ax.set_title(f"{title_prefix}: {task_filter} Tasks", fontsize=16, fontweight='bold')
    ax.set_xlabel("PC1 (Semantic Variance)")
    ax.set_ylabel("PC2 (Semantic Variance)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Clinical Diagnosis", fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.show()

def generate_single_file_embeddings(file_number, data_dir='data'):
    """
    Processes a specific file by its number (e.g., '097') and generates embeddings.
    """
    loader = TranscriptLoader(data_dir=Path(data_dir))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Discovers all files and filter for the specific number
    transcript_paths = loader.discover_transcript_files()
    target_paths = [p for p in transcript_paths if f"rec-{file_number}" in p.name]
    
    if not target_paths:
        print(f"No files found for recording number: {file_number}")
        return pd.DataFrame()

    all_data = []
    print(f"Processing {len(target_paths)} file(s) for recording {file_number}...")

    for path in target_paths:
        df = loader.load_transcript(path)
        
        # Extracts base ID and merges with metadata
        recording_id = path.stem.split('__')[0].strip()
        df['recording'] = recording_id 
        merged_df = loader.merge_with_metadata(df)
        
        if merged_df.empty:
            print(f"Warning: No metadata found for {recording_id}")
            continue
            
        # Standardise the diagnosises and generate embeddings
        raw_diag = str(merged_df['diagnosis'].iloc[0]).strip().upper()
        diagnosis = 'Dementia' if 'DEMENTIA' in raw_diag else 'MCI' if 'MCI' in raw_diag else 'HC' if 'HC' in raw_diag else 'Unknown'

        tokens = df['token'].dropna().astype(str).str.lower().tolist()
        if not tokens: continue
            
        embeddings = model.encode(tokens)
        task_type = loader.infer_task_from_filename(path.name)
        
        for token, emb in zip(tokens, embeddings):
            all_data.append({
                'recording': recording_id,
                'task': task_type,
                'token': token,
                'diagnosis': diagnosis, 
                'embedding': emb
            })

    return pd.DataFrame(all_data)


def train_lr_classifier(df, k_folds=6, n_components=20, task_mode='Fused', binary_mode=True):
    """
    Implements Logistic Regression with Task Separation (SFT vs PFT vs Fused).
    Supports Strategy B (PCA) and Strategy C (Binary: HC vs Dementia).
    """
    recordings = df['recording'].unique()
    processed_data = []

    # FEATURE EXTRACTION LOOP
    for rid in recordings:
        patient_subset = df[df['recording'] == rid]
        
        # Filter by task based on the mode
        sft_sub = patient_subset[patient_subset['task'] == 'SFT'].sort_values('start_time')
        pft_sub = patient_subset[patient_subset['task'] == 'PFT'].sort_values('start_time')
        
        # Skip if the required task data is missing
        if (task_mode == 'Fused' and (sft_sub.empty or pft_sub.empty)): continue
        if (task_mode == 'SFT' and sft_sub.empty): continue
        if (task_mode == 'PFT' and pft_sub.empty): continue

        # EXTRACT ONLY SFT FEATURES
        if not sft_sub.empty:
            sft_mean = np.mean(np.stack(sft_sub['embedding'].values), axis=0)
            s_pause = (sft_sub['start_time'].shift(-1) - sft_sub['end_time']).mean()
            sft_feats = np.concatenate([sft_mean, [len(sft_sub), s_pause if not np.isnan(s_pause) else 0]])

        # EXTRACT ONLY PFT FEATURES
        if not pft_sub.empty:
            pft_mean = np.mean(np.stack(pft_sub['embedding'].values), axis=0)
            p_pause = (pft_sub['start_time'].shift(-1) - pft_sub['end_time']).mean()
            pft_feats = np.concatenate([pft_mean, [len(pft_sub), p_pause if not np.isnan(p_pause) else 0]])

        # FUSION OF BOTH OR SINGLE TASK
        if task_mode == 'Fused':
            final_vector = np.concatenate([sft_feats, pft_feats])
        elif task_mode == 'SFT':
            final_vector = sft_feats
        else: # PFT
            final_vector = pft_feats
        
        processed_data.append({
            'features': np.nan_to_num(final_vector), 
            'diagnosis': patient_subset['diagnosis'].iloc[0]
        })

    final_df = pd.DataFrame(processed_data)

    # Exclude MCI if in binary mode (2 way classification)
    if binary_mode:
        final_df = final_df[final_df['diagnosis'].isin(['HC', 'Dementia'])].copy()
    
    X = np.array(final_df['features'].tolist())
    y = np.array(final_df['diagnosis'].tolist())

    # K-FOLD PREDICTIONS: Logistic Regression Pipeline
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Using Logistic Regression with Balanced weights
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    all_y_true, all_y_pred = [], []

    print(f"Validating {task_mode} LR ({'Binary' if binary_mode else '3-Class'}) across {k_folds} folds...")
    for train_idx, test_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        all_y_pred.extend(pipeline.predict(X[test_idx]))
        all_y_true.extend(y[test_idx])

    print(f"\n=== Final {task_mode} LR Results (All {len(X)} Participants) ===")
    print(f"Overall Accuracy: {accuracy_score(all_y_true, all_y_pred):.2%}")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

    plot_fused_confusion_matrix(all_y_true, all_y_pred, title=f"LR: {task_mode} Mode")

    pipeline.fit(X, y)
    return pipeline

def plot_fused_confusion_matrix(y_true, y_pred, title="Fused SVM Confusion Matrix"):
    """
    Plots a heatmap that dynamically detects class labels to avoid 
    label mismatch in binary vs multiclass modes.
    """
    # Detects unique labels present in the data and sorts them 
    # This ensures 'Dementia' and 'HC' appear in the correct order.
    class_names = sorted(list(set(y_true))) 
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Calculate percentages for each row (True Positive)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    labels = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm)):
            row.append(f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)")
        labels.append(row)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.show()

if __name__ == "__main__":
    all_results = generate_diagnosis_embeddings()
    visualise_by_diagnosis(all_results, task_filter='SFT')