"""
Compressed Latent Memory Graph (CLMG)
Replaces KV-cache with associative episodic memory.
Fixed memory budget regardless of sequence length.
"""
import torch
import numpy as np
import faiss
import sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *

class CompressedLatentMemoryGraph:
    def __init__(self):
        self.node_dim = CLMG_NODE_DIM
        self.max_nodes = CLMG_MAX_NODES
        self.top_k = CLMG_TOP_K
        self.decay = CLMG_DECAY_RATE

        # Compressor: 512 → 64
        self.compressor = torch.nn.Sequential(
            torch.nn.Linear(HWM_LEVEL_DIMS[0], 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, self.node_dim),
            torch.nn.LayerNorm(self.node_dim)
        ).to(DEVICE).to(DTYPE)

        # Decompressor: 64 → 512
        self.decompressor = torch.nn.Sequential(
            torch.nn.Linear(self.node_dim, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, HWM_LEVEL_DIMS[0])
        ).to(DEVICE).to(DTYPE)

        # FAISS index for fast similarity search (CPU — saves VRAM)
        self.index = faiss.IndexFlatL2(self.node_dim)
        self.nodes = []          # list of compressed float32 vectors
        self.activation = []     # retrieval frequency per node
        self.node_count = 0

    def store(self, state: torch.Tensor):
        """Compress and store a world state. state: (B, 512)"""
        with torch.no_grad():
            compressed = self.compressor(state)  # (B, 64)
            compressed_np = compressed.float().cpu().numpy()

        for vec in compressed_np:
            if self.node_count >= self.max_nodes:
                self._prune()
            self.index.add(vec.reshape(1, -1))
            self.nodes.append(vec)
            self.activation.append(1.0)
            self.node_count += 1

    def retrieve(self, query: torch.Tensor):
        """Find top-K most relevant memories. query: (B, 512)"""
        if self.node_count == 0:
            return torch.zeros(
                query.shape[0], self.top_k, HWM_LEVEL_DIMS[0],
                device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            q_compressed = self.compressor(query).float().cpu().numpy()

        k = min(self.top_k, self.node_count)
        results = []
        for q in q_compressed:
            D, I = self.index.search(q.reshape(1, -1), k)
            retrieved = np.stack([self.nodes[i] for i in I[0]])  # (k, 64)
            # Boost activation counts
            for i in I[0]: self.activation[i] += 1.0
            # Decompress
            t = torch.tensor(retrieved, device=DEVICE, dtype=DTYPE)
            decompressed = self.decompressor(t)  # (k, 512)
            # Pad if fewer than top_k
            if k < self.top_k:
                pad = torch.zeros(self.top_k - k, HWM_LEVEL_DIMS[0],
                                  device=DEVICE, dtype=DTYPE)
                decompressed = torch.cat([decompressed, pad], dim=0)
            results.append(decompressed)

        return torch.stack(results)  # (B, top_k, 512)

    def _prune(self):
        """Remove least-activated 10% of nodes"""
        n_prune = max(1, self.node_count // 10)
        scores = np.array(self.activation)
        # Apply decay
        self.activation = [a * (1 - self.decay) for a in self.activation]
        drop_idx = set(np.argsort(scores)[:n_prune])
        keep_idx = [i for i in range(self.node_count) if i not in drop_idx]
        kept_nodes = [self.nodes[i] for i in keep_idx]
        kept_act   = [self.activation[i] for i in keep_idx]
        # Rebuild FAISS index
        self.index.reset()
        self.nodes = []
        self.activation = []
        self.node_count = 0
        for vec, act in zip(kept_nodes, kept_act):
            self.index.add(vec.reshape(1, -1))
            self.nodes.append(vec)
            self.activation.append(act)
            self.node_count += 1

    def stats(self):
        return {
            "nodes": self.node_count,
            "max_nodes": self.max_nodes,
            "fill_pct": self.node_count / self.max_nodes * 100
        }

if __name__ == "__main__":
    print("Building Compressed Latent Memory Graph...")
    mem = CompressedLatentMemoryGraph()

    # Store 100 random world states
    for i in range(100):
        state = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
        mem.store(state)

    print(f"Stored: {mem.stats()}")

    # Retrieve
    query = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
    retrieved = mem.retrieve(query)
    print(f"Query:     {query.shape}")
    print(f"Retrieved: {retrieved.shape}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print("✓ CLMG operational")
