"""
M4: Explainability Module

Evaluates the reasoning transparency of AI-generated content:
- Reasoning chain analysis
- Attribution detection
- Hedging language identification
- Confidence qualification assessment
"""

import re
from typing import Dict, Any, List


class ExplainabilityEngine:
    """
    Analyzes explainability and reasoning transparency
    in AI-generated content.
    
    Evaluates:
    - Presence of reasoning chains
    - Source attribution
    - Uncertainty acknowledgment
    - Logical consistency
    """
    
    # Positive explainability indicators
    REASONING_INDICATORS = [
        (r"\b(because|since|therefore|thus|hence|as a result)\b", "causal_reasoning", 0.15),
        (r"\b(first|second|third|finally|additionally|moreover)\b", "structured_reasoning", 0.1),
        (r"\b(for example|for instance|such as|e\.g\.|i\.e\.)\b", "example_provision", 0.12),
        (r"\b(according to|based on|research shows|studies indicate)\b", "source_attribution", 0.15),
        (r"\b(this means|in other words|to clarify|specifically)\b", "clarification", 0.1),
    ]
    
    # Hedging language (positive - shows appropriate uncertainty)
    HEDGING_INDICATORS = [
        (r"\b(might|may|could|possibly|perhaps|likely|probably)\b", "uncertainty_hedge", 0.1),
        (r"\b(suggests?|appears?|seems?|indicates?)\b", "tentative_language", 0.08),
        (r"\b(in some cases|sometimes|often|generally|typically)\b", "qualified_claim", 0.1),
        (r"\b(approximately|about|around|roughly|estimated)\b", "precision_qualifier", 0.08),
    ]
    
    # Negative indicators (reduce explainability)
    OPACITY_INDICATORS = [
        (r"\b(obviously|clearly|undeniably|unquestionably)\b", "overconfidence", -0.15),
        (r"\b(everyone knows|it's common knowledge|as we all know)\b", "assumed_knowledge", -0.12),
        (r"\b(trust me|believe me|take my word)\b", "appeal_to_trust", -0.2),
        (r"\b(always|never|all|none|every)\b", "absolute_claim", -0.08),
    ]
    
    # Logical connectors (positive structure)
    LOGIC_CONNECTORS = [
        "however", "although", "nevertheless", "on the other hand",
        "in contrast", "while", "whereas", "despite", "conversely"
    ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze explainability of the text.
        
        Args:
            text: Content to analyze
        
        Returns:
            Dictionary with score, verdict, and findings
        """
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        word_count = len(text.split())
        
        positive_score = 0.0
        negative_score = 0.0
        findings = {
            "reasoning_indicators": [],
            "hedging_indicators": [],
            "opacity_indicators": [],
            "structure_analysis": {}
        }
        
        # 1. Check reasoning indicators
        for pattern, indicator_type, weight in self.REASONING_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                positive_score += weight * min(len(matches), 5)
                findings["reasoning_indicators"].append({
                    "type": indicator_type,
                    "count": len(matches),
                    "contribution": round(weight * min(len(matches), 5), 3)
                })
        
        # 2. Check hedging (appropriate uncertainty)
        for pattern, indicator_type, weight in self.HEDGING_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                positive_score += weight * min(len(matches), 4)
                findings["hedging_indicators"].append({
                    "type": indicator_type,
                    "count": len(matches),
                    "contribution": round(weight * min(len(matches), 4), 3)
                })
        
        # 3. Check opacity indicators (negative)
        for pattern, indicator_type, weight in self.OPACITY_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                negative_score += abs(weight) * min(len(matches), 3)
                findings["opacity_indicators"].append({
                    "type": indicator_type,
                    "count": len(matches),
                    "penalty": round(abs(weight) * min(len(matches), 3), 3)
                })
        
        # 4. Structural analysis
        logic_connector_count = sum(1 for c in self.LOGIC_CONNECTORS if c in text_lower)
        avg_sentence_length = word_count / max(len(sentences), 1)
        has_paragraphs = "\n\n" in text or text.count("\n") > 2
        
        structure_score = 0.0
        if logic_connector_count >= 2:
            structure_score += 0.15
        if 15 <= avg_sentence_length <= 30:
            structure_score += 0.1
        if has_paragraphs:
            structure_score += 0.1
        if len(sentences) >= 3:
            structure_score += 0.1
        
        positive_score += structure_score
        
        findings["structure_analysis"] = {
            "logic_connectors": logic_connector_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "sentence_count": len(sentences),
            "has_paragraphs": has_paragraphs,
            "structure_score": round(structure_score, 3)
        }
        
        # 5. Calculate final score
        raw_score = positive_score - negative_score
        
        # Normalize to 0-100 scale
        # Expected range: -0.5 to 1.5, normalize accordingly
        normalized = (raw_score + 0.5) / 2.0  # Maps to 0-1
        score = max(0, min(100, round(normalized * 100)))
        
        # Adjust based on content length
        if word_count < 50:
            score = min(score, 70)  # Short content has limited explainability opportunity
        
        # 6. Determine verdict
        if score >= 75:
            verdict = "Explainable — Clear Reasoning"
        elif score >= 50:
            verdict = "Partially Explainable"
        elif score >= 30:
            verdict = "Opaque — Limited Transparency"
        else:
            verdict = "Inconsistent — Poor Explainability"
        
        findings_summary = self._generate_summary(findings, score)
        
        return {
            "score": score,
            "verdict": verdict,
            "findings": findings_summary,
            "details": {
                "positive_indicators": len(findings["reasoning_indicators"]) + len(findings["hedging_indicators"]),
                "negative_indicators": len(findings["opacity_indicators"]),
                "structure": findings["structure_analysis"],
                "raw_scores": {
                    "positive": round(positive_score, 3),
                    "negative": round(negative_score, 3),
                    "net": round(raw_score, 3)
                }
            }
        }
    
    def _generate_summary(self, findings: Dict, score: int) -> str:
        """Generate human-readable summary."""
        parts = []
        
        reasoning_count = len(findings["reasoning_indicators"])
        hedging_count = len(findings["hedging_indicators"])
        opacity_count = len(findings["opacity_indicators"])
        
        if reasoning_count > 0:
            parts.append(f"{reasoning_count} reasoning indicator(s) found")
        
        if hedging_count > 0:
            parts.append(f"appropriate uncertainty language present")
        
        if opacity_count > 0:
            parts.append(f"{opacity_count} opacity concern(s)")
        
        structure = findings["structure_analysis"]
        if structure.get("logic_connectors", 0) >= 2:
            parts.append("logical structure detected")
        
        if not parts:
            if score >= 60:
                return "Content shows reasonable transparency with structured reasoning."
            else:
                return "Limited explainability indicators. Consider adding reasoning chains and source attribution."
        
        return "; ".join(parts).capitalize() + "."
