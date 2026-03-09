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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for label, color in color_map.items():
        mask = subset['diagnosis'] == label
        if not mask.any(): 
            continue # Skip if a group has no words to prevent legend errors
        
        ax.scatter(
            reduced_vecs[mask, 0], 
            reduced_vecs[mask, 1], 
            c=color, 
            label=label, 
            alpha=0.9,         # High opacity
            s=80,              # Larger dots
            edgecolors='black', # High contrast border
            linewidth=0.8
        )
    
    ax.set_title(f"Linguistic Semantic Map: {task_filter} Tasks", fontsize=16, fontweight='bold')
    ax.set_xlabel("PC1 (Semantic Variance)")
    ax.set_ylabel("PC2 (Semantic Variance)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Clinical Diagnosis", fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_df = generate_diagnosis_embeddings()
    visualise_by_diagnosis(results_df, task_filter='SFT')