"""
Neuro-Symbolic Abstraction Layer (NSAL)
Proper nn.Module so dtype/device conversion works correctly.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict
import json, sys, os
sys.path.append(os.path.expanduser("~/spe"))
from core.config import *

@dataclass
class Rule:
    id: int
    description: str
    confidence: float
    antecedents: List[int]
    embedding: List[float]
    activation_count: int = 0
    verified: bool = False

class CodeGraph:
    def __init__(self):
        self.rules: Dict[int, Rule] = {}
        self.next_id = 0
        self.edges: Dict[int, List[int]] = {}

    def add_rule(self, description, confidence, antecedents, embedding):
        rule = Rule(id=self.next_id, description=description,
                    confidence=confidence, antecedents=antecedents,
                    embedding=embedding)
        self.rules[self.next_id] = rule
        self.edges[self.next_id] = []
        for ant in antecedents:
            if ant in self.edges:
                self.edges[ant].append(self.next_id)
        self.next_id += 1
        return rule.id

    def rule_count(self): return len(self.rules)

    def prune_weak(self, min_confidence=0.3):
        to_remove=[rid for rid,r in self.rules.items()
                   if r.confidence<min_confidence and not r.verified]
        for rid in to_remove:
            del self.rules[rid]; del self.edges[rid]
        return len(to_remove)

    def save(self, path):
        data={str(k):{"id":r.id,"description":r.description,
                      "confidence":r.confidence,"antecedents":r.antecedents,
                      "activation_count":r.activation_count,"verified":r.verified}
              for k,r in self.rules.items()}
        with open(path,'w') as f: json.dump(data,f,indent=2)


class NSAL(nn.Module):
    """
    Neuro-Symbolic Abstraction Layer — proper nn.Module.
    Extracts symbolic rules from world model states.
    """
    def __init__(self, n_rule_slots=32):
        super().__init__()
        self.n_rule_slots = n_rule_slots
        dim_in = HWM_LEVEL_DIMS[2] + HWM_LEVEL_DIMS[3]  # 256+128=384

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )
        self.rule_heads = nn.ModuleList([
            nn.Linear(128, CLMG_NODE_DIM + 1)
            for _ in range(n_rule_slots)
        ])
        self.code_graph = CodeGraph()
        self.confidence_threshold = NSAL_CONFIDENCE_THRESHOLD

    def process(self, world_states: list, context: str = ""):
        level3 = world_states[2]  # (B, 256)
        level4 = world_states[3]  # (B, 128)
        x = torch.cat([level3, level4], dim=-1)
        x = self.encoder(x)
        embeddings, confidences = [], []
        for head in self.rule_heads:
            out = head(x)
            embeddings.append(out[..., :-1])
            confidences.append(torch.sigmoid(out[..., -1:]))
        embeddings = torch.stack(embeddings, dim=1)
        confidences = torch.cat(confidences, dim=-1)

        # Promote high-confidence patterns (no_grad — symbolic side)
        with torch.no_grad():
            B = confidences.shape[0]
            new_rules = 0
            for b in range(B):
                for slot in range(confidences.shape[1]):
                    conf = confidences[b, slot].item()
                    if conf >= self.confidence_threshold:
                        emb = embeddings[b, slot].cpu().float().tolist()
                        self.code_graph.add_rule(
                            description=f"rule_{self.code_graph.next_id}",
                            confidence=conf, antecedents=[], embedding=emb)
                        new_rules += 1
                        if self.code_graph.rule_count() >= NSAL_MAX_RULES:
                            self.code_graph.prune_weak()

        return {
            "new_rules": new_rules,
            "total_rules": self.code_graph.rule_count(),
            "mean_confidence": confidences.mean().item(),
            "embeddings": embeddings,
            "confidences": confidences,
        }

    def save(self, path): self.code_graph.save(path)
