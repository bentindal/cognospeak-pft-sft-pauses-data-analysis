import re
import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
from .dataloader import TranscriptLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

keras = tf.keras
layers = tf.keras.layers

loader = TranscriptLoader()
transcript_paths = loader.discover_transcript_files()

pft_paths = [p for p in transcript_paths if loader.infer_task_from_filename(p.name) == "PFT"]
sft_paths = [p for p in transcript_paths if loader.infer_task_from_filename(p.name) == "SFT"]

dataframes_p = {p.stem: loader.load_transcript(p) for p in pft_paths}
dataframes_s = {p.stem: loader.load_transcript(p) for p in sft_paths}

print("PFT files:", len(dataframes_p))
print("SFT files:", len(dataframes_s))

# print(dataframes_p)

file_vocab, index = {}, 1  
file_inverse_vocab = {}
file_sequences = {}

for fname, df in dataframes_p.items():
    tokens = df["token"].dropna().astype(str).str.lower().unique()
    vocab = {"<pad>": 0, "<unk>": 1}

    for idx, tok in enumerate(tokens,start=2):
        vocab[tok] = idx

    file_vocab[fname] = vocab
    file_inverse_vocab[fname] = {i: t for t, i in vocab.items()}
    file_vocab_size = len(vocab)
    file_inverse_vocab_size = len(file_inverse_vocab)
    seq = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    file_sequences[fname] = seq

first_file = next(iter(file_sequences))
print("First file:", first_file)
print("First 30 ids:", file_sequences[first_file][:30])
print("Vocab size:", len(file_vocab[first_file]))

##### BUILDING GLOBAL VOCAB ######


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


#     return vocab, inverse_vocab, vocab_size, sequences
def build_global_vocab_and_sequences(dfs: dict[str, pd.DataFrame], col: str = "token"):
    all_words = pd.concat((df[col] for df in dfs.values()), ignore_index=True)
    all_words = all_words.dropna().astype("string").str.lower().str.strip()

    unique_tokens = all_words.unique()
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({tok: i for i, tok in enumerate(unique_tokens, start=2)})

    inverse_vocab = {i: t for t, i in vocab.items()}
    vocab_size = len(vocab)

    unk_id = vocab["<unk>"]
    sequences = {
        fname: df[col].dropna().astype("string").str.lower().str.strip()
                  .map(lambda w: vocab.get(w, unk_id)).to_list()
        for fname, df in dfs.items()
    }
    return vocab, inverse_vocab, vocab_size, sequences


vocab_p, inv_vocab_p, vocab_size_p, seqs_p = build_global_vocab_and_sequences(dataframes_p)
vocab_s, inv_vocab_s, vocab_size_s, seqs_s = build_global_vocab_and_sequences(dataframes_s)


# Word2Vec helpers
def make_skipgram_training_data(seqs: dict[str, list[int]],
                               vocab_size: int,
                               window_size: int = 2,
                               num_negative: int = 5,
                               seed: int = 42,
                               max_pairs: int | None = None):
    rng = np.random.default_rng(seed)

    targets_list = []
    contexts_list = []

    # positive (target, context) pairs (skip-gram)
    for _, seq in seqs.items():
        n = len(seq)
        if n < 2:
            continue

        for i, target in enumerate(seq):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)

            for j in range(left, right):
                if j == i:
                    continue
                targets_list.append(target)
                contexts_list.append(seq[j])

                if max_pairs is not None and len(targets_list) >= max_pairs:
                    break
            if max_pairs is not None and len(targets_list) >= max_pairs:
                break
        if max_pairs is not None and len(targets_list) >= max_pairs:
            break

    if len(targets_list) == 0:
        raise ValueError("No skip-gram pairs were generated. Check your sequences/window_size.")

    targets = np.asarray(targets_list, dtype=np.int32)
    true_context = np.asarray(contexts_list, dtype=np.int32).reshape(-1, 1)  # (N, 1)

    N = len(targets)
    print("Positive pairs:", N)
    print("\n")

    # per-example negative sampling: (N, K)
    # sample from [2, vocab_size) to avoid <pad>=0 and <unk>=1
    negative_context = rng.integers(2, vocab_size, size=(N, num_negative), dtype=np.int32)

    # make sure negatives are not equal to the true context
    mask = (negative_context == true_context)
    while mask.any():
        negative_context[mask] = rng.integers(2, vocab_size, size=mask.sum(), dtype=np.int32)
        mask = (negative_context == true_context)

    # pack contexts and labels
    context_all = np.concatenate([true_context, negative_context], axis=1)  # (N, 1+K)

    labels = np.zeros((N, 1 + num_negative), dtype=np.float32)
    labels[:, 0] = 1.0  # first column corresponds to the true context

    return targets, context_all, labels


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.target_emb = layers.Embedding(vocab_size, embedding_dim, name="target_embedding")
        self.context_emb = layers.Embedding(vocab_size, embedding_dim, name="context_embedding")

    def call(self, inputs):
        target, context = inputs  # target: (B,), context: (B, 1+K)
        v = self.target_emb(target)      # (B, D)
        u = self.context_emb(context)    # (B, 1+K, D)
        logits = tf.einsum("bd,bkd->bk", v, u)  # dot products -> (B, 1+K)
        return logits


def train_word2vec(seqs, vocab_size, window_size=2, num_negative=3,
                   embedding_dim=32, batch_size=1024, epochs=10,
                   max_pairs=None, seed=42):
    targets, context_all, labels = make_skipgram_training_data(
        seqs=seqs,
        vocab_size=vocab_size,
        window_size=window_size,
        num_negative=num_negative,
        seed=seed,
        max_pairs=max_pairs
    )

    ds = tf.data.Dataset.from_tensor_slices(((targets, context_all), labels))
    ds = ds.shuffle(min(200_000, len(targets)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
    )

    model.fit(ds, epochs=epochs)
    return model


def most_similar(word: str, vocab: dict[str, int], inverse_vocab: dict[int, str],
                 vectors: np.ndarray, top_k: int = 10):
    wid = vocab.get(word, vocab.get("<unk>", 1))
    v = vectors[wid]
    v = v / (np.linalg.norm(v) + 1e-9)

    W = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    sims = W @ v

    best = np.argsort(-sims)[:top_k + 1]  # includes itself
    out = [(inverse_vocab[i], float(sims[i])) for i in best if i != wid][:top_k]
    return out


# Train PFT Word2Vec
w2v_p = train_word2vec(
    seqs=seqs_p,
    vocab_size=vocab_size_p,
    window_size=2,
    num_negative=3,
    embedding_dim=32,
    batch_size=32,
    epochs=10,
    max_pairs=300_000,   # set None to use all pairs; cap prevents RAM blowups
    seed=SEED
)

vectors_p = w2v_p.get_layer("target_embedding").get_weights()[0]

print(most_similar("president", vocab_p, inv_vocab_p, vectors_p, top_k=10))
print("\n")
# Train SFT Word2Vec
w2v_s = train_word2vec(
    seqs=seqs_s,
    vocab_size=vocab_size_s,
    window_size=2,
    num_negative=3,
    embedding_dim=32,
    batch_size=32,
    epochs=10,
    max_pairs=300_000,
    seed=SEED
)

vectors_s = w2v_s.get_layer("target_embedding").get_weights()[0]

print(most_similar("aardvark", vocab_s, inv_vocab_s, vectors_s, top_k=10))

print("\n")

### pca ###

def plot_embeddings_pca(vectors, vocab, inv_vocab, words=None, top_n=200, title="Embeddings (PCA)"):
    # which words to plot
    if words is None:
        # default: plot first top_n tokens excluding <pad>/<unk>
        ids = [i for w,i in vocab.items() if w not in ("<pad>", "<unk>")]
        ids = sorted(ids)[:top_n]
    else:
        ids = [vocab[w] for w in words if w in vocab]
        if len(ids) == 0:
            raise ValueError("None of the requested words were found in the vocab.")

    X = vectors[ids]  # (n_words, dim)

    # PCA to 2D
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(X2[:, 0], X2[:, 1], s=25)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Label only a few points to avoid a mess
    max_labels = min(30, len(ids))
    for k in range(max_labels):
        wid = ids[k]
        plt.text(X2[k, 0], X2[k, 1], inv_vocab[wid], fontsize=9)

    plt.show()

# Example: SFT
plot_embeddings_pca(vectors_s, vocab_s, inv_vocab_s, top_n=200, title="SFT Word2Vec (PCA)")

# Example: PFT
plot_embeddings_pca(vectors_p, vocab_p, inv_vocab_p, top_n=200, title="PFT Word2Vec (PCA)")

### TSNE ###

def plot_embeddings_tsne(vectors, vocab, inv_vocab, words=None, top_n=300,
                         perplexity=30, title="Embeddings (t-SNE)"):
    if words is None:
        ids = [i for w,i in vocab.items() if w not in ("<pad>", "<unk>")]
        ids = sorted(ids)[:top_n]
    else:
        ids = [vocab[w] for w in words if w in vocab]
        if len(ids) == 0:
            raise ValueError("None of the requested words were found in the vocab.")

    X = vectors[ids].astype(np.float32)

    # t-SNE to 2D
    X2 = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(perplexity, max(5, len(ids)//3)),
        random_state=42
    ).fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(X2[:, 0], X2[:, 1], s=25)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    max_labels = min(40, len(ids))
    for k in range(max_labels):
        wid = ids[k]
        plt.text(X2[k, 0], X2[k, 1], inv_vocab[wid], fontsize=9)

    plt.show()

# Example: SFT
plot_embeddings_tsne(vectors_s, vocab_s, inv_vocab_s, top_n=300, perplexity=25, title="SFT Word2Vec (t-SNE)")

# Example: PFT
plot_embeddings_tsne(vectors_p, vocab_p, inv_vocab_p, top_n=300, perplexity=25, title="PFT Word2Vec (t-SNE)")