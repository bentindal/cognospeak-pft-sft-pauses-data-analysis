import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from dataloader import TranscriptLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def generate_diagnosis_embeddings(data_dir='data'):
    loader = TranscriptLoader(data_dir=Path(data_dir))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    transcript_paths = loader.discover_transcript_files()
    all_data = []

    print(f"Processing {len(transcript_paths)} files...")
    for path in transcript_paths:
        df = loader.load_transcript(path)
        
        # Extract base ID to match CSV
        recording_id = path.stem.split('__')[0].strip()
        df['recording'] = recording_id 
        
        merged_df = loader.merge_with_metadata(df)
        if merged_df.empty:
            continue
            
        raw_diag = str(merged_df['diagnosis'].iloc[0]).strip().upper()
        if 'DEMENTIA' in raw_diag: diagnosis = 'Dementia'
        elif 'MCI' in raw_diag: diagnosis = 'MCI'
        elif 'HC' in raw_diag: diagnosis = 'HC'
        else: continue

        # Filter out empty tokens
        df = df.dropna(subset=['token'])
        tokens = df['token'].astype(str).str.lower().tolist()
        if not tokens: continue
            
        # Extract timing data
        start_times = df['start_time'].tolist()
        end_times = df['end_time'].tolist()
        
        embeddings = model.encode(tokens)
        task_type = loader.infer_task_from_filename(path.name)
        
        # FIX: Save start_time and end_time for the classifier to use later
        for i in range(len(tokens)):
            all_data.append({
                'recording': recording_id,
                'task': task_type,
                'token': tokens[i],
                'start_time': start_times[i], # Needed for pause calculation
                'end_time': end_times[i],     # Needed for pause calculation
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
    
    # 1. Plot the dots
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
    
    # 2. Add labels ONLY if it's a single file
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
    
    # 1. Discover all files and filter for the specific number
    transcript_paths = loader.discover_transcript_files()
    target_paths = [p for p in transcript_paths if f"rec-{file_number}" in p.name]
    
    if not target_paths:
        print(f"No files found for recording number: {file_number}")
        return pd.DataFrame()

    all_data = []
    print(f"Processing {len(target_paths)} file(s) for recording {file_number}...")

    for path in target_paths:
        df = loader.load_transcript(path)
        
        # 2. Extract base ID and merge with metadata
        recording_id = path.stem.split('__')[0].strip()
        df['recording'] = recording_id 
        merged_df = loader.merge_with_metadata(df)
        
        if merged_df.empty:
            print(f"Warning: No metadata found for {recording_id}")
            continue
            
        # 3. Standardize diagnosis and generate embeddings
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

def train_fusion_svm_classifier(df):
    """
    Implements Early Fusion: Concatenates SFT and PFT features into a single 
    vector to train a Support Vector Machine (SVM).
    """
    recordings = df['recording'].unique()
    fused_patient_data = []

    print(f"Performing Early Fusion for {len(recordings)} participants...")

    for rid in recordings:
        # Isolate tasks for this specific participant
        patient_subset = df[df['recording'] == rid]
        sft_sub = patient_subset[patient_subset['task'] == 'SFT'].sort_values('start_time')
        pft_sub = patient_subset[patient_subset['task'] == 'PFT'].sort_values('start_time')
        
        # Only fuse if the participant has completed BOTH tasks
        if sft_sub.empty or pft_sub.empty:
            continue
            
        # 1. SFT FEATURES (Meaning + Volume + Hesitation)
        sft_mean_emb = np.mean(np.stack(sft_sub['embedding'].values), axis=0)
        sft_count = len(sft_sub)
        sft_pauses = sft_sub['start_time'].shift(-1) - sft_sub['end_time']
        sft_mean_pause = sft_pauses.dropna().mean() if not sft_pauses.dropna().empty else 0
        
        # 2. PFT FEATURES (Meaning + Volume + Hesitation)
        pft_mean_emb = np.mean(np.stack(pft_sub['embedding'].values), axis=0)
        pft_count = len(pft_sub)
        pft_pauses = pft_sub['start_time'].shift(-1) - pft_sub['end_time']
        pft_mean_pause = pft_pauses.dropna().mean() if not pft_pauses.dropna().empty else 0
        
        # THE FUSION: Glue them all together into vector X
        # Length: 384 (SFT) + 384 (PFT) + 4 (Temporal) = 772 dimensions
        fusion_vector = np.concatenate([
            sft_mean_emb, 
            pft_mean_emb, 
            [sft_count, sft_mean_pause, pft_count, pft_mean_pause]
        ])
        
        fused_patient_data.append({
            'features': fusion_vector,
            'diagnosis': sft_sub['diagnosis'].iloc[0]
        })

    if not fused_patient_data:
        print("Error: No participants found with both SFT and PFT data.")
        return None

    final_df = pd.DataFrame(fused_patient_data)
    X = np.array(final_df['features'].tolist())
    y = np.array(final_df['diagnosis'].tolist())

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. SVM CLASSIFIER: Optimal for high-dimensional fused vectors
    # We use a linear kernel first as it is less likely to overfit small samples
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)
    
    print(f"\n=== Early Fusion SVM Results (SFT + PFT) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return svm_model

if __name__ == "__main__":
    all_results = generate_diagnosis_embeddings()
    fusion_model = train_fusion_svm_classifier(all_results)
    visualise_by_diagnosis(all_results, task_filter='SFT')