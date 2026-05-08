"""
M5: Deepfake Detection Module (Text-Based)

For text content, this module detects AI-generation markers:
- Repetitive patterns
- Unnatural phrasing
- Statistical anomalies
- Style consistency issues

Note: For image/video content, this would integrate with
actual deepfake detection models (ELA, DCT, FaceForensics++).
"""

import re
from typing import Dict, Any, List
from collections import Counter
import math


class DeepfakeDetector:
    """
    Detects synthetic/AI-generated content markers.
    
    For text: Analyzes patterns typical of AI generation
    For images: Would use ELA, DCT, and neural detection
    """
    
    # AI-generation markers in text
    AI_MARKERS = [
        # Overly formal/corporate language
        (r"\b(leverage|utilize|implement|facilitate|optimize)\b", "corporate_speak", 0.05),
        # Common AI phrases
        (r"\b(it's important to note|it's worth mentioning|in conclusion)\b", "ai_phrase", 0.1),
        (r"\b(as an ai|as a language model|i don't have personal)\b", "ai_disclosure", 0.8),
        # Hedging pile-up
        (r"(may|might|could)\s+(potentially|possibly)\s+\w+", "hedge_pileup", 0.15),
        # Perfect structure
        (r"^(first|1\.)\s.*^(second|2\.)\s.*^(third|3\.)\s", "perfect_enumeration", 0.1),
    ]
    
    # Natural language indicators (reduce AI suspicion)
    HUMAN_MARKERS = [
        # Informal language
        (r"\b(gonna|wanna|kinda|sorta|yeah|nope)\b", "informal", -0.1),
        # Contractions
        (r"\b(can't|won't|don't|isn't|aren't|wasn't)\b", "contractions", -0.05),
        # First person narrative
        (r"\b(I think|I believe|I feel|in my opinion|personally)\b", "personal_voice", -0.08),
        # Emotional expression
        (r"\b(love|hate|excited|frustrated|amazing|terrible)\b", "emotional", -0.05),
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.compiled_ai = [(re.compile(p, re.IGNORECASE | re.MULTILINE), d, w) 
                            for p, d, w in self.AI_MARKERS]
        self.compiled_human = [(re.compile(p, re.IGNORECASE), d, w) 
                               for p, d, w in self.HUMAN_MARKERS]
    
    def detect(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Detect synthetic content markers.
        
        Args:
            text: Content to analyze
            content_type: Type of content
        
        Returns:
            Dictionary with score, verdict, and findings
        """
        findings = []
        ai_score = 0.0
        human_score = 0.0
        
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)
        
        # 1. Check AI markers
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
        
        # 2. Check human markers
        for pattern, marker_type, weight in self.compiled_human:
            matches = pattern.findall(text)
            if matches:
                human_score += abs(weight) * len(matches)
                findings.append({
                    "type": "human_marker",
                    "marker": marker_type,
                    "count": len(matches)
                })
        
        # 3. Statistical analysis
        stats = self._statistical_analysis(text, words)
        findings.append({
            "type": "statistical_analysis",
            "details": stats
        })
        
        # Repetitive patterns suggest AI
        if stats["repetition_ratio"] > 0.1:
            ai_score += 0.2
        
        # Very uniform sentence length is AI-like
        if stats["sentence_length_variance"] < 3:
            ai_score += 0.15
        
        # 4. Vocabulary analysis
        vocab_analysis = self._vocabulary_analysis(words)
        if vocab_analysis["unique_ratio"] < 0.4:
            ai_score += 0.1  # Low vocabulary diversity
        
        # Content type adjustments
        if content_type == "academic":
            ai_score *= 0.8  # Academic writing is naturally more formal
        elif content_type == "social":
            # Social media should have more human markers
            if human_score < 0.2:
                ai_score += 0.15
        
        # 5. Calculate final score (higher = more authentic = better)
        net_score = human_score - ai_score
        
        # Normalize: expected range -1 to +0.5
        normalized = (net_score + 1) / 1.5
        score = max(0, min(100, round(normalized * 100)))
        
        # Adjust for edge cases
        if "ai_disclosure" in [f.get("marker") for f in findings]:
            score = min(score, 20)  # AI self-disclosure = clearly AI
        
        # 6. Determine verdict
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
        """Perform statistical analysis on text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sentence length analysis
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
        
        if len(sentence_lengths) > 1:
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
        else:
            variance = 0
            std_dev = 0
        
        # Repetition analysis
        word_counts = Counter(words)
        repeated_words = sum(1 for w, c in word_counts.items() if c > 2 and len(w) > 4)
        repetition_ratio = repeated_words / max(len(set(words)), 1)
        
        # Phrase repetition
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
        """Analyze vocabulary diversity."""
        total = len(words)
        unique = len(set(w.lower() for w in words))
        
        # Calculate type-token ratio
        ttr = unique / max(total, 1)
        
        # Long word ratio (>7 characters)
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
        """Generate human-readable summary."""
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
