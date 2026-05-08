"""
Integrity Index Aggregator

Computes the final Integrity Index using weighted harmonic mean
across all five modules.
"""

from typing import Dict, Any
from config import config


class IntegrityAggregator:
    """
    Aggregates module scores into the Integrity Index.
    
    Uses weighted harmonic mean to ensure that a single
    failing dimension significantly impacts the overall score.
    """
    
    def __init__(self):
        self.weights = config.WEIGHTS
    
    def compute_integrity_index(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute the Integrity Index from module scores.
        
        Formula: II = Σwᵢ / Σ(wᵢ / Sᵢ)
        
        Args:
            scores: Dictionary mapping module names to scores (0-100)
        
        Returns:
            Dictionary with final score and trust tier
        """
        # Ensure all modules are present
        required_modules = ["m1_bias", "m2_hallucination", "m3_privacy", 
                          "m4_explainability", "m5_deepfake"]
        
        for module in required_modules:
            if module not in scores:
                scores[module] = 50  # Default neutral score
        
        # Calculate weighted harmonic mean
        numerator = sum(self.weights.values())
        
        denominator = 0.0
        for module, weight in self.weights.items():
            score = max(scores.get(module, 50), 1)  # Avoid division by zero
            denominator += weight / score
        
        if denominator > 0:
            integrity_index = numerator / denominator
        else:
            integrity_index = 0
        
        # Round to one decimal place
        integrity_index = round(integrity_index, 1)
        
        # Determine trust tier
        if integrity_index >= 85:
            tier = "High Trust"
            tier_description = "Content meets high integrity standards across all dimensions."
        elif integrity_index >= 65:
            tier = "Moderate Trust"
            tier_description = "Content has some integrity concerns but is generally acceptable."
        elif integrity_index >= 40:
            tier = "Low Trust"
            tier_description = "Significant integrity issues detected. Use with caution."
        else:
            tier = "Reject"
            tier_description = "Critical integrity failures. Content should not be trusted."
        
        return {
            "score": integrity_index,
            "tier": tier,
            "tier_description": tier_description,
            "module_contributions": self._calculate_contributions(scores),
            "formula": "Weighted Harmonic Mean"
        }
    
    def _calculate_contributions(self, scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Calculate how each module contributes to the final score."""
        contributions = {}
        
        for module, weight in self.weights.items():
            score = scores.get(module, 50)
            contributions[module] = {
                "score": score,
                "weight": weight,
                "weighted_score": round(score * weight, 2),
                "impact": "positive" if score >= 70 else "neutral" if score >= 40 else "negative"
            }
        
        return contributions
