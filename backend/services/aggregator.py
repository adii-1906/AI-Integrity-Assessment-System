"""
Integrity Index Aggregator
Same class IntegrityAggregator, same method compute_integrity_index().
Works without config import - uses hardcoded weights.
"""

from typing import Dict, Any


class IntegrityAggregator:
    """Same interface as original. No config dependency."""

    WEIGHTS = {
        "m1_bias": 0.20,
        "m2_hallucination": 0.25,
        "m3_privacy": 0.20,
        "m4_explainability": 0.15,
        "m5_deepfake": 0.20
    }

    def __init__(self):
        # Try to import config; fall back to hardcoded weights
        try:
            from config import config
            self.weights = config.WEIGHTS
        except Exception:
            self.weights = self.WEIGHTS

    def compute_integrity_index(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Same signature and return structure as original.
        Formula: II = Σwi / Σ(wi / Si)
        """
        required_modules = ["m1_bias", "m2_hallucination", "m3_privacy",
                            "m4_explainability", "m5_deepfake"]
        for module in required_modules:
            if module not in scores:
                scores[module] = 50

        numerator = sum(self.weights.values())
        denominator = 0.0
        for module, weight in self.weights.items():
            score = max(scores.get(module, 50), 1)
            denominator += weight / score

        integrity_index = round(numerator / denominator, 1) if denominator > 0 else 0

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