"""
M5: Deepfake / Synthetic Text Detection Module — ML VERSION
Loads trained Gradient Boosting + TF-IDF from train_models.py.
Feature matrix = TF-IDF (15000) hstacked with 7 statistical features
(same as train_models.py: TTR, sentence length variance, bigram repeat,
word count, unique word count, long word ratio, avg sentence length).

Falls back to rule-based if models not yet trained.

Model files needed (run train_models.py first):
    backend/models/m5_deepfake_model.pkl
    backend/models/m5_deepfake_tfidf.pkl
"""

import os
import re
import math
import pickle
import logging
import numpy as np
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_BASE_DIR, "..", "models")

MODEL_PATH  = os.path.join(_MODEL_DIR, "m5_deepfake_model.pkl")
TFIDF_PATH  = os.path.join(_MODEL_DIR, "m5_deepfake_tfidf.pkl")


# ── Same 7 statistical features as in train_models.py ─────────────────
def _extract_statistical_features(text: str) -> List[float]:
    """Exactly matches extract_statistical_features() in train_models.py."""
    if not text or len(text) < 10:
        return [0.5, 5.0, 0.0, 50, 30, 0.0, 0.3]

    words     = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # 1. Type-Token Ratio
    ttr = len(set(words)) / len(words) if words else 0.5

    # 2. Sentence length variance
    sent_lengths = [len(s.split()) for s in sentences]
    variance     = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 5.0

    # 3. Bigram repetition ratio
    bigrams      = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_rep   = ((len(bigrams) - len(set(bigrams))) / len(bigrams)) if bigrams else 0.0

    # 4. Total word count
    word_count   = len(words)

    # 5. Unique word count
    unique_count = len(set(words))

    # 6. Long word ratio (words > 7 chars)
    long_ratio   = sum(1 for w in words if len(w) > 7) / len(words) if words else 0.0

    # 7. Average sentence length
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 15.0

    return [ttr, variance, bigram_rep, word_count, unique_count, long_ratio, avg_sent_len]


class DeepfakeDetector:
    """
    M5 Deepfake / Synthetic Text Detection.
    Primary: Gradient Boosting on TF-IDF + 7 statistical features (HC3 + GLTR).
    Supplementary: HC3-derived lexical markers (always run).
    Fallback: lexical + statistical rule-based only.
    Same interface: detect(text, content_type) → dict.
    """

    # ── AI markers (HC3 analysis — supplementary layer) ───────────────
    _AI_MARKERS = [
        (r"\b(leverage|utilize|implement|facilitate|optimize)\b",             "corporate_speak",  0.05),
        (r"\b(it's important to note|in conclusion|in summary|delve into)\b", "ai_phrase",        0.10),
        (r"\b(as an ai|as a language model|i don't have personal)\b",         "ai_disclosure",    0.80),
        (r"(may|might|could)\s+(potentially|possibly)\s+\w+",                 "hedge_pileup",     0.15),
        (r"\b(holistic approach|robust solution|seamless experience)\b",       "ai_platitude",     0.07),
        (r"\b(first and foremost|last but not least|with that being said)\b",  "ai_transition",    0.07),
    ]

    # ── Human markers (HC3 human responses — supplementary layer) ──────
    _HUMAN_MARKERS = [
        (r"\b(gonna|wanna|kinda|sorta|yeah|nope|yep)\b", "informal",       -0.10),
        (r"\b(can't|won't|don't|isn't|aren't|didn't)\b",  "contractions",  -0.05),
        (r"\b(I think|I believe|to be honest|tbh)\b",     "personal_voice",-0.08),
        (r"\b(love|hate|excited|frustrated|amazing|lol)\b","emotional",     -0.05),
    ]

    # Content-type probability adjustment
    _CONTENT_ADJ = {"academic": -0.08, "social": +0.10, "news": 0.0, "text": 0.0}

    def __init__(self):
        self._model      = None
        self._tfidf      = None
        self._ml_ready   = False
        self._load_models()
        self._compile_patterns()

    # ── Model loading ──────────────────────────────────────────────────
    def _load_models(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(TFIDF_PATH, "rb") as f:
                    self._tfidf = pickle.load(f)
                self._ml_ready = True
                print("  [M5 Deepfake] ML model loaded ✓")
            except Exception as e:
                print(f"  [M5 Deepfake] WARNING: model load failed ({e}) — using rule-based fallback")
        else:
            print("  [M5 Deepfake] WARNING: trained models not found — run train_models.py first. Using rule-based fallback.")

    def _compile_patterns(self):
        self._ai_compiled    = [(re.compile(p, re.IGNORECASE), t, w) for p, t, w in self._AI_MARKERS]
        self._human_compiled = [(re.compile(p, re.IGNORECASE), t, w) for p, t, w in self._HUMAN_MARKERS]

    # ── Primary method ─────────────────────────────────────────────────
    def detect(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        findings      = []
        adjustment    = self._CONTENT_ADJ.get(content_type, 0.0)

        # ── 1.  ML inference (Gradient Boosting) ──────────────────────
        if self._ml_ready:
            ml_ai_prob = self._ml_predict(text)                     # probability text is AI-generated
            ml_ai_prob = max(0.0, min(1.0, ml_ai_prob + adjustment))
            score_ml   = max(0, round((1.0 - ml_ai_prob) * 100))    # higher score = more human
            ai_disclosure = bool(re.search(
                r'\b(as an ai|as a language model)\b', text, re.IGNORECASE))
            if ai_disclosure:
                score_ml = min(score_ml, 20)
            findings.append({
                "type":  "ml_classifier",
                "model": "Gradient Boosting (HC3 TF-IDF + 7 statistical features)",
                "ai_probability": round(ml_ai_prob, 4),
                "severity": ml_ai_prob,
            })
        else:
            score_ml   = None
            ml_ai_prob = None

        # ── 2.  Supplementary lexical layer (always runs) ─────────────
        ai_score    = 0.0
        human_score = 0.0

        for pattern, marker, weight in self._ai_compiled:
            matches = pattern.findall(text)
            if matches:
                ai_score += weight * len(matches)
                findings.append({"type": "ai_marker", "marker": marker, "count": len(matches)})

        for pattern, marker, weight in self._human_compiled:
            matches = pattern.findall(text)
            if matches:
                human_score += abs(weight) * len(matches)
                findings.append({"type": "human_marker", "marker": marker, "count": len(matches)})

        # ── 3.  Statistical features (GLTR) ───────────────────────────
        stats = _extract_statistical_features(text)
        stat_names = ["ttr", "sent_variance", "bigram_repeat",
                      "word_count", "unique_count", "long_ratio", "avg_sent_len"]
        stat_dict  = dict(zip(stat_names, stats))
        findings.append({"type": "statistical", "details": stat_dict})

        # Rule-based score from lexical + statistical
        # Matches original formula: net_score = human_score - ai_score
        if stats[1] < 3.0: ai_score    += 0.15   # low sentence variance
        if stats[0] < 0.40: ai_score   += 0.10   # low TTR
        if stats[2] > 0.10: ai_score   += 0.20   # high bigram repeat

        net          = human_score - ai_score
        normalised   = (net + 1) / 1.5
        score_rule   = max(0, min(100, round(normalised * 100)))

        ai_disclosure_rule = any(
            f.get("marker") == "ai_disclosure"
            for f in findings if f.get("type") == "ai_marker"
        )
        if ai_disclosure_rule:
            score_rule = min(score_rule, 20)

        # ── 4.  Blend ─────────────────────────────────────────────────
        if self._ml_ready:
            # 70 % ML, 30 % rule
            score = round(0.70 * score_ml + 0.30 * score_rule)
            if ai_disclosure:
                score = min(score, 20)
        else:
            score = score_rule

        # ── 5.  Verdict ───────────────────────────────────────────────
        if score >= 75:
            verdict = "Authentic — Human-like Content"
        elif score >= 50:
            verdict = "Suspicious — Mixed Indicators"
        elif score >= 25:
            verdict = "Likely Synthetic — AI Markers Present"
        else:
            verdict = "Synthetic — Strong AI Generation Indicators"

        return {
            "score":   score,
            "verdict": verdict,
            "findings": self._summary(findings, score),
            "details": {
                "ml_used":     self._ml_ready,
                "model":       "Gradient Boosting (HC3 + GLTR)" if self._ml_ready else "Rule-based fallback",
                "score_ml":    score_ml,
                "score_rule":  score_rule,
                "ai_markers":  len([f for f in findings if f.get("type") == "ai_marker"]),
                "human_markers": len([f for f in findings if f.get("type") == "human_marker"]),
                "statistics":  stat_dict,
                "raw_scores":  {"ai": round(ai_score, 3), "human": round(human_score, 3)},
            }
        }

    # ── ML inference ───────────────────────────────────────────────────
    def _ml_predict(self, text: str) -> float:
        """
        Build the SAME feature matrix as train_models.py:
        hstack([X_tfidf, csr_matrix(X_stats)])
        Then call model.predict_proba().
        """
        from scipy.sparse import hstack, csr_matrix

        X_tfidf = self._tfidf.transform([text])          # sparse (1, 15000)
        X_stats = np.array([_extract_statistical_features(text)])
        X_comb  = hstack([X_tfidf, csr_matrix(X_stats)]) # sparse (1, 15007)

        prob = self._model.predict_proba(X_comb.toarray())[0]
        # Class 1 = AI-generated
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def _summary(self, findings: List[Dict], score: int) -> str:
        parts = []
        ml_f  = [f for f in findings if f.get("type") == "ml_classifier"]
        if ml_f:
            prob = ml_f[0]["ai_probability"]
            parts.append(f"ML AI-generation probability: {prob:.0%}")
        ai_m  = [f for f in findings if f.get("type") == "ai_marker"]
        hm_m  = [f for f in findings if f.get("type") == "human_marker"]
        if ai_m:
            types = set(f.get("marker") for f in ai_m)
            parts.append(f"AI markers: {', '.join(types)}")
        if hm_m:
            parts.append("human language indicators present")
        return ("; ".join(parts).capitalize() + ".") if parts else "Analysis complete."