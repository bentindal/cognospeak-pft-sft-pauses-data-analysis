import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from dataloader import TranscriptLoader

def generate_diagnosis_embeddings(data_dir='data'):
    """
    Generates LLM embeddings and maps them to clinical diagnosis labels.
    """
    loader = TranscriptLoader(data_dir=Path(data_dir))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    transcript_paths = loader.discover_transcript_files()
    all_data = []

    print(f"Processing {len(transcript_paths)} files with clinical labels...")
    for path in transcript_paths:
        df = loader.load_transcript(path)
        
        # 1. Manually add the 'recording' column to match the file name
        # path.stem will be 'Process-rec-097', which matches 'anyon_IDs' in your CSV
        df['recording'] = path.stem 
        
        task_type = loader.infer_task_from_filename(path.name)
        
        # 2. Merge with metadata
        # The loader internally handles mapping 'anyon_IDs' to 'recording'
        merged_df = loader.merge_with_metadata(df)
        
        if merged_df.empty:
            print(f"Warning: No metadata found for {path.stem}")
            continue
            
        # 3. Standardize diagnosis labels (Dementia, MCI, HC)
        # Note: Your CSV uses 'Dementia ', 'MCI      ', so we strip whitespace
        diagnosis = str(merged_df['diagnosis'].iloc[0]).strip()
        
        tokens = df['token'].dropna().astype(str).str.lower().tolist()
        if not tokens:
            continue
            
        embeddings = model.encode(tokens)
        
        for token, emb in zip(tokens, embeddings):
            all_data.append({
                'recording': path.stem,
                'task': task_type,
                'token': token,
                'diagnosis': diagnosis, 
                'embedding': emb
            })

    return pd.DataFrame(all_data)

def visualise_by_diagnosis(df, task_filter='SFT'):
    """
    Visualizes the semantic space using high-contrast colors for clinical groups.
    """
    subset = df[df['task'] == task_filter].copy()
    
    # 1. Use the project's official bold color palette
    # Red for Dementia, Orange for MCI, and Green for HC
    color_map = {
        'Dementia': '#d62728', 
        'MCI': '#ff7f0e', 
        'HC': '#2ca02c'
    }
    
    if subset.empty:
        print(f"No data for {task_filter}")
        return
        
    embeddings = np.stack(subset['embedding'].values)
    pca = PCA(n_components=2, random_state=42)
    reduced_vecs = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 2. Plot groups with higher alpha (opacity) and larger dots for visibility
    for label, color in color_map.items():
        mask = subset['diagnosis'] == label
        if not mask.any(): continue
        
        ax.scatter(
            reduced_vecs[mask, 0], 
            reduced_vecs[mask, 1], 
            c=color, 
            label=label, 
            alpha=0.8,        # Increased from 0.5 for better visibility
            s=60,             # Larger marker size
            edgecolors='black', # Black edges make colors pop
            linewidth=0.5
        )
    
    # 3. Formatting for professional presentation
    ax.set_title(f"Semantic 'Meaning-Space' by Clinical Diagnosis ({task_filter})", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Principal Component 1 (Major Semantic Variance)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (Minor Semantic Variance)", fontsize=12)
    
    # Force axis ticks to be clearly visible
    ax.tick_params(axis='both', which='major', labelsize=10, bottom=True, left=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Bold Legend
    legend = ax.legend(title="Diagnosis Group", frameon=True, fontsize=11)
    plt.setp(legend.get_title(), fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_df = generate_diagnosis_embeddings()
    visualise_by_diagnosis(results_df, task_filter='SFT')