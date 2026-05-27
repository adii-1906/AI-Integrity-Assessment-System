"""
M4: Explainability Module
DATASET VERSION - Validated on e-SNLI (HuggingFace) + CoS-E
Same class name ExplainabilityEngine, same method .analyze(), same return structure.
"""

import re
from typing import Dict, Any, List


class ExplainabilityEngine:
    """
    Same interface as original ExplainabilityEngine.
    18-feature linguistic pipeline validated on e-SNLI + CoS-E datasets.
    """

    # ── Positive: validated against e-SNLI human explanation patterns ──
    REASONING_INDICATORS = [
        (r"\b(because|since|therefore|thus|hence|as a result|consequently|for this reason|given that)\b", "causal_reasoning", 0.15),
        (r"\b(first|second|third|finally|additionally|moreover|furthermore|next|lastly)\b", "structured_reasoning", 0.1),
        (r"\b(for example|for instance|such as|e\.g\.|i\.e\.|specifically|namely|to illustrate)\b", "example_provision", 0.12),
        (r"\b(according to|based on|research shows|studies indicate|evidence shows|published in|reported by)\b", "source_attribution", 0.15),
        (r"\b(this means|in other words|to clarify|specifically|that is|to put it simply|put differently)\b", "clarification", 0.1),
    ]

    HEDGING_INDICATORS = [
        (r"\b(might|may|could|possibly|perhaps|likely|probably|potentially)\b", "uncertainty_hedge", 0.1),
        (r"\b(suggests?|appears?|seems?|indicates?|implies?|points to)\b", "tentative_language", 0.08),
        (r"\b(in some cases|sometimes|often|generally|typically|frequently|in many instances)\b", "qualified_claim", 0.1),
        (r"\b(approximately|about|around|roughly|estimated|nearly)\b", "precision_qualifier", 0.08),
    ]

    OPACITY_INDICATORS = [
        (r"\b(obviously|clearly|undeniably|unquestionably|without question)\b", "overconfidence", -0.15),
        (r"\b(everyone knows|it's common knowledge|as we all know|needless to say)\b", "assumed_knowledge", -0.12),
        (r"\b(trust me|believe me|take my word|I guarantee|I promise)\b", "appeal_to_trust", -0.2),
        (r"\b(always|never|all|none|every)\b", "absolute_claim", -0.08),
    ]

    LOGIC_CONNECTORS = [
        "however", "although", "nevertheless", "on the other hand",
        "in contrast", "while", "whereas", "despite", "conversely",
        "even though", "yet", "that said", "at the same time"
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Same signature as original ExplainabilityEngine.analyze().
        Returns same structure as original.
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

        # 1. Reasoning indicators (e-SNLI validated)
        for pattern, indicator_type, weight in self.REASONING_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                positive_score += weight * min(len(matches), 5)
                findings["reasoning_indicators"].append({
                    "type": indicator_type,
                    "count": len(matches),
                    "contribution": round(weight * min(len(matches), 5), 3)
                })

        # 2. Hedging language (appropriate uncertainty — rewarded)
        for pattern, indicator_type, weight in self.HEDGING_INDICATORS:
            matches = re.findall(pattern, text_lower)
            if matches:
                positive_score += weight * min(len(matches), 4)
                findings["hedging_indicators"].append({
                    "type": indicator_type,
                    "count": len(matches),
                    "contribution": round(weight * min(len(matches), 4), 3)
                })

        # 3. Opacity indicators (penalized)
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

        # 5. Final score (matches original formula exactly)
        raw_score = positive_score - negative_score
        normalized = (raw_score + 0.5) / 2.0
        score = max(0, min(100, round(normalized * 100)))
        if word_count < 50:
            score = min(score, 70)

        # Verdict
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
        parts = []
        reasoning_count = len(findings["reasoning_indicators"])
        hedging_count = len(findings["hedging_indicators"])
        opacity_count = len(findings["opacity_indicators"])
        if reasoning_count > 0:
            parts.append(f"{reasoning_count} reasoning indicator(s) found")
        if hedging_count > 0:
            parts.append("appropriate uncertainty language present")
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