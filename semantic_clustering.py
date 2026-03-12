import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from dataloader import TranscriptLoader

def generate_diagnosis_embeddings(data_dir='data'):
    loader = TranscriptLoader(data_dir=Path(data_dir))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    transcript_paths = loader.discover_transcript_files()
    all_data = []

    print(f"Processing {len(transcript_paths)} files...")
    for path in transcript_paths:
        df = loader.load_transcript(path)
        
        # FIX: Extract only the base ID (e.g., 'Process-rec-097') 
        # This removes the '__PFT' or '__SFT' suffix to match your CSV
        recording_id = path.stem.split('__')[0].strip()
        df['recording'] = recording_id 
        
        # Merge using your existing dataloader logic
        merged_df = loader.merge_with_metadata(df)
        
        if merged_df.empty:
            continue
            
        # Clean and standardize diagnosis labels from your CSV
        raw_diag = str(merged_df['diagnosis'].iloc[0]).strip().upper()
        if 'DEMENTIA' in raw_diag: diagnosis = 'Dementia'
        elif 'MCI' in raw_diag: diagnosis = 'MCI'
        elif 'HC' in raw_diag: diagnosis = 'HC'
        else: continue

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

    # Return valid DataFrame or structure with columns to avoid KeyErrors
    if not all_data:
        return pd.DataFrame(columns=['recording', 'task', 'token', 'diagnosis', 'embedding'])
        
    return pd.DataFrame(all_data)

def visualise_by_diagnosis(df, task_filter='SFT'):
    """
    Visualizes the semantic space. 
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

if __name__ == "__main__":
    results_df = generate_diagnosis_embeddings()
    visualise_by_diagnosis(results_df, task_filter='SFT')