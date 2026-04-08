"""
Neuro-Symbolic Abstraction Layer (NSAL)
Extracts logical rules from world model states.
Rules are stored in a mutable Code-Graph and used for inference.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json, sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *

@dataclass
class Rule:
    id: int
    description: str
    confidence: float
    antecedents: List[int]   # rule IDs that must hold
    embedding: List[float]   # latent representation
    activation_count: int = 0
    verified: bool = False

class CodeGraph:
    """Mutable directed graph of learned symbolic rules."""
    def __init__(self):
        self.rules: Dict[int, Rule] = {}
        self.next_id = 0
        self.edges: Dict[int, List[int]] = {}  # rule_id → dependent rule_ids

    def add_rule(self, description, confidence, antecedents, embedding):
        rule = Rule(
            id=self.next_id,
            description=description,
            confidence=confidence,
            antecedents=antecedents,
            embedding=embedding
        )
        self.rules[self.next_id] = rule
        self.edges[self.next_id] = []
        for ant in antecedents:
            if ant in self.edges:
                self.edges[ant].append(self.next_id)
        self.next_id += 1
        return rule.id

    def get_rule(self, rule_id): return self.rules.get(rule_id)
    def rule_count(self): return len(self.rules)

    def prune_weak(self, min_confidence=0.3):
        """Remove rules below confidence threshold."""
        to_remove = [rid for rid, r in self.rules.items()
                     if r.confidence < min_confidence and not r.verified]
        for rid in to_remove:
            del self.rules[rid]
            del self.edges[rid]
        return len(to_remove)

    def save(self, path):
        data = {str(k): {
            "id": r.id, "description": r.description,
            "confidence": r.confidence, "antecedents": r.antecedents,
            "activation_count": r.activation_count, "verified": r.verified
        } for k, r in self.rules.items()}
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        print(f"Saved {len(data)} rules to {path}")


class RuleExtractor(nn.Module):
    """
    Extracts candidate rules from world model states.
    Input: level-3 (relational/causal) and level-4 (abstract) states
    Output: rule embeddings + confidence scores
    """
    def __init__(self, n_rule_slots=32):
        super().__init__()
        self.n_rule_slots = n_rule_slots
        dim_in = HWM_LEVEL_DIMS[2] + HWM_LEVEL_DIMS[3]  # 256+128=384

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )
        # Each slot: one candidate rule
        self.rule_heads = nn.ModuleList([
            nn.Linear(128, CLMG_NODE_DIM + 1)  # embedding + confidence
            for _ in range(n_rule_slots)
        ])

    def forward(self, level3_state, level4_state):
        """
        level3: (B, 256), level4: (B, 128)
        Returns: embeddings (B, n_slots, 64), confidences (B, n_slots)
        """
        x = torch.cat([level3_state, level4_state], dim=-1)
        x = self.encoder(x)

        embeddings, confidences = [], []
        for head in self.rule_heads:
            out = head(x)
            embeddings.append(out[..., :-1])
            confidences.append(torch.sigmoid(out[..., -1:]))

        embeddings   = torch.stack(embeddings, dim=1)   # (B, slots, 64)
        confidences  = torch.cat(confidences, dim=-1)   # (B, slots)
        return embeddings, confidences


class NSAL:
    """Full Neuro-Symbolic Abstraction Layer."""
    def __init__(self):
        self.extractor = RuleExtractor().to(DEVICE).to(DTYPE)
        self.code_graph = CodeGraph()
        self.confidence_threshold = NSAL_CONFIDENCE_THRESHOLD
        self.pending_patterns: List[Dict] = []

    def process(self, world_states: list, context: str = ""):
        """
        Extract rules from world model states.
        world_states: list of 4 tensors from HierarchicalWorldModel
        """
        level3 = world_states[2]  # (B, 256)
        level4 = world_states[3]  # (B, 128)

        with torch.no_grad():
            embeddings, confidences = self.extractor(level3, level4)

        # Promote high-confidence patterns to rules
        new_rules = 0
        B = confidences.shape[0]
        for b in range(B):
            for slot in range(confidences.shape[1]):
                conf = confidences[b, slot].item()
                if conf >= self.confidence_threshold:
                    emb = embeddings[b, slot].cpu().tolist()
                    rid = self.code_graph.add_rule(
                        description=f"auto_rule_{self.code_graph.next_id}",
                        confidence=conf,
                        antecedents=[],
                        embedding=emb
                    )
                    new_rules += 1
                    if self.code_graph.rule_count() >= NSAL_MAX_RULES:
                        self.code_graph.prune_weak()

        return {
            "new_rules": new_rules,
            "total_rules": self.code_graph.rule_count(),
            "mean_confidence": confidences.mean().item()
        }

    def infer(self, query_embedding: torch.Tensor):
        """
        Find rules most similar to query embedding.
        Returns top matching rules by cosine similarity.
        """
        if self.code_graph.rule_count() == 0:
            return []
        q = query_embedding.float().cpu().numpy()
        results = []
        for rule in self.code_graph.rules.values():
            emb = torch.tensor(rule.embedding)
            sim = torch.nn.functional.cosine_similarity(
                query_embedding.float().cpu(),
                emb.unsqueeze(0), dim=-1).item()
            results.append((sim, rule))
        results.sort(key=lambda x: -x[0])
        return results[:NSAL_MAX_RULE_DEPTH]

    def save(self, path): self.code_graph.save(path)


if __name__ == "__main__":
    print("Building NSAL...")
    nsal = NSAL()
    params = sum(p.numel() for p in nsal.extractor.parameters())
    print(f"Extractor parameters: {params:,}")

    # Simulate world states
    world_states = [
        torch.randn(4, 512, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 384, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 256, device=DEVICE, dtype=DTYPE),
        torch.randn(4, 128, device=DEVICE, dtype=DTYPE),
    ]

    for i in range(5):
        result = nsal.process(world_states)

    print(f"Rules extracted: {result}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print("✓ NSAL operational")
