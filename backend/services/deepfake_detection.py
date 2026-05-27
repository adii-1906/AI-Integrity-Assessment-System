"""
M5: Deepfake Detection Module
DATASET VERSION - HC3 (HuggingFace) + GLTR Research
Same class name DeepfakeDetector, same method .detect(), same return structure.
"""

import re
from typing import Dict, Any, List
from collections import Counter
import math


class DeepfakeDetector:
    """
    Same interface as original DeepfakeDetector.
    HC3 dataset lexicons + GLTR statistical features. No API.
    """

    # ── AI markers (from HC3 ChatGPT vs Human analysis) ──
    AI_MARKERS = [
        (r"\b(leverage|utilize|implement|facilitate|optimize)\b", "corporate_speak", 0.05),
        (r"\b(it's important to note|it's worth mentioning|in conclusion|in summary|to summarize)\b", "ai_phrase", 0.1),
        (r"\b(as an ai|as a language model|i don't have personal|as an ai assistant)\b", "ai_disclosure", 0.8),
        (r"(may|might|could)\s+(potentially|possibly)\s+\w+", "hedge_pileup", 0.15),
        (r"^(first|1\.)\s.*^(second|2\.)\s.*^(third|3\.)\s", "perfect_enumeration", 0.1),
        (r"\b(in the realm of|it is essential to|in today's|delve into|dive into)\b", "ai_phrase", 0.08),
        (r"\b(a holistic approach|robust solution|seamless experience|comprehensive overview)\b", "ai_phrase", 0.07),
        (r"\b(first and foremost|last but not least|at the end of the day|with that being said)\b", "ai_phrase", 0.07),
        (r"\b(navigate the complexities|rest assured|needless to say)\b", "ai_phrase", 0.07),
    ]

    # ── Human markers (from HC3 human responses) ──
    HUMAN_MARKERS = [
        (r"\b(gonna|wanna|kinda|sorta|yeah|nope|yep)\b", "informal", -0.1),
        (r"\b(can't|won't|don't|isn't|aren't|wasn't|didn't)\b", "contractions", -0.05),
        (r"\b(I think|I believe|I feel|in my opinion|personally|to be honest|tbh)\b", "personal_voice", -0.08),
        (r"\b(love|hate|excited|frustrated|amazing|terrible|honestly|lol|omg)\b", "emotional", -0.05),
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled_ai = [(re.compile(p, re.IGNORECASE | re.MULTILINE), d, w)
                            for p, d, w in self.AI_MARKERS]
        self.compiled_human = [(re.compile(p, re.IGNORECASE), d, w)
                               for p, d, w in self.HUMAN_MARKERS]

    def detect(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Same signature as original DeepfakeDetector.detect().
        Returns same structure as original.
        """
        findings = []
        ai_score = 0.0
        human_score = 0.0

        words = text.split()
        word_count = len(words)

        # 1. AI marker detection (HC3 derived)
        for pattern, marker_type, weight in self.compiled_ai:
            matches = pattern.findall(text)
            if matches:
                ai_score += weight * len(matches)
                findings.append({
                    "type": "ai_marker",
                    "marker": marker_type,
                    "count": len(matches),
                    "weight": weight
                })

        # 2. Human marker detection (HC3 derived)
        for pattern, marker_type, weight in self.compiled_human:
            matches = pattern.findall(text)
            if matches:
                human_score += abs(weight) * len(matches)
                findings.append({
                    "type": "human_marker",
                    "marker": marker_type,
                    "count": len(matches)
                })

        # 3. GLTR statistical analysis
        stats = self._statistical_analysis(text, words)
        findings.append({"type": "statistical_analysis", "details": stats})

        if stats["repetition_ratio"] > 0.1:
            ai_score += 0.2
        if stats["sentence_length_variance"] < 3:
            ai_score += 0.15

        # 4. Vocabulary diversity (GLTR TTR feature)
        vocab_analysis = self._vocabulary_analysis(words)
        if vocab_analysis["unique_ratio"] < 0.4:
            ai_score += 0.1

        # 5. Content type adjustments (matches original logic)
        if content_type == "academic":
            ai_score *= 0.8
        elif content_type == "social":
            if human_score < 0.2:
                ai_score += 0.15

        # 6. Final score (matches original formula exactly)
        net_score = human_score - ai_score
        normalized = (net_score + 1) / 1.5
        score = max(0, min(100, round(normalized * 100)))

        if "ai_disclosure" in [f.get("marker") for f in findings]:
            score = min(score, 20)

        # Verdict
        if score >= 75:
            verdict = "Authentic — Human-like Content"
        elif score >= 50:
            verdict = "Suspicious — Mixed Indicators"
        elif score >= 25:
            verdict = "Likely Synthetic — AI Markers Present"
        else:
            verdict = "Synthetic — Strong AI Generation Indicators"

        findings_summary = self._generate_summary(findings, ai_score, human_score, stats)

        return {
            "score": score,
            "verdict": verdict,
            "findings": findings_summary,
            "details": {
                "ai_markers_found": len([f for f in findings if f["type"] == "ai_marker"]),
                "human_markers_found": len([f for f in findings if f["type"] == "human_marker"]),
                "statistics": stats,
                "vocabulary": vocab_analysis,
                "raw_scores": {
                    "ai_score": round(ai_score, 3),
                    "human_score": round(human_score, 3)
                }
            }
        }

    def _statistical_analysis(self, text: str, words: List[str]) -> Dict[str, Any]:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
        if len(sentence_lengths) > 1:
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
        else:
            variance = 0
            std_dev = 0
        word_counts = Counter(words)
        repeated_words = sum(1 for w, c in word_counts.items() if c > 2 and len(w) > 4)
        repetition_ratio = repeated_words / max(len(set(words)), 1)
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for b, c in bigram_counts.items() if c > 1)
        return {
            "avg_sentence_length": round(avg_length, 2),
            "sentence_length_variance": round(variance, 2),
            "sentence_length_std": round(std_dev, 2),
            "sentence_count": len(sentences),
            "repetition_ratio": round(repetition_ratio, 3),
            "repeated_bigrams": repeated_bigrams
        }

    def _vocabulary_analysis(self, words: List[str]) -> Dict[str, Any]:
        total = len(words)
        unique = len(set(w.lower() for w in words))
        ttr = unique / max(total, 1)
        long_words = sum(1 for w in words if len(w) > 7)
        long_ratio = long_words / max(total, 1)
        return {
            "total_words": total,
            "unique_words": unique,
            "unique_ratio": round(ttr, 3),
            "long_word_ratio": round(long_ratio, 3)
        }

    def _generate_summary(self, findings: List[Dict], ai_score: float,
                          human_score: float, stats: Dict) -> str:
        parts = []
        ai_markers = [f for f in findings if f["type"] == "ai_marker"]
        human_markers = [f for f in findings if f["type"] == "human_marker"]
        if ai_score > human_score:
            if ai_score > 0.5:
                parts.append("Strong AI-generation indicators detected")
            else:
                parts.append("Moderate AI-generation patterns present")
            marker_types = [f.get("marker") for f in ai_markers]
            if "ai_phrase" in marker_types:
                parts.append("common AI phrases found")
            if "corporate_speak" in marker_types:
                parts.append("corporate/formal language patterns")
        else:
            parts.append("Content appears human-authored")
            if human_markers:
                parts.append("natural language indicators present")
        if stats.get("repetition_ratio", 0) > 0.1:
            parts.append("notable phrase repetition")
        return "; ".join(parts).capitalize() + "." if parts else "Analysis complete."