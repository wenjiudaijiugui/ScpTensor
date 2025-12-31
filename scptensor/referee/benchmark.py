from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from scptensor.core.structures import ScpContainer
from .metrics import compute_batch_mixing, compute_cluster_separation, compute_ari

class IntegrationReferee:
    def __init__(self, container: ScpContainer, batch_key: str, label_key: str = None):
        """
        Referee for evaluating integration methods.
        
        Args:
            container: ScpContainer.
            batch_key: Column name in obs for batch info.
            label_key: Column name in obs for biological labels (optional).
        """
        self.container = container
        self.batch_key = batch_key
        self.label_key = label_key

    def score(self, assay_name: str, layer: str = 'X') -> dict:
        """
        Calculate scores for a specific Assay/Layer.
        """
        if assay_name not in self.container.assays:
             raise ValueError(f"Assay '{assay_name}' not found.")
        
        assay = self.container.assays[assay_name]
        if layer not in assay.layers:
             raise ValueError(f"Layer '{layer}' not found in assay '{assay_name}'.")

        X = assay.layers[layer].X
        batches = self.container.obs[self.batch_key].to_numpy()
        
        scores = {}
        
        # 1. Batch Mixing (Higher is better)
        scores['batch_mixing_score'] = compute_batch_mixing(X, batches)
        
        # 2. Bio Conservation (if labels available)
        if self.label_key and self.label_key in self.container.obs.columns:
            labels = self.container.obs[self.label_key].to_numpy()
            scores['bio_conservation_score'] = compute_cluster_separation(X, labels)
            
        return scores

    def compare(self, candidates: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Compare multiple results.
        
        Args:
            candidates: List of (assay_name, layer_name) tuples.
            
        Returns:
            pd.DataFrame with scores.
        """
        results = []
        for assay, layer in candidates:
            try:
                s = self.score(assay, layer)
                s['method'] = f"{assay}_{layer}"
                results.append(s)
            except ValueError as e:
                print(f"Skipping {assay}_{layer}: {e}")
                
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['method'] + [c for c in df.columns if c != 'method']
        return df[cols]
