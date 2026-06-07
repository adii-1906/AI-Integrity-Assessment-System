"""
M1: Bias Detection Module — ML VERSION
Loads trained Logistic Regression + TF-IDF from train_models.py output.
Falls back to rule-based if models not yet trained.

Model files needed (run train_models.py first):
    backend/models/m1_bias_model.pkl
    backend/models/m1_bias_vectorizer.pkl
"""

import os
import re
import pickle
import logging
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "..", "models")

MODEL_PATH      = os.path.join(_MODEL_DIR, "m1_bias_model.pkl")
VECTORIZER_PATH = os.path.join(_MODEL_DIR, "m1_bias_vectorizer.pkl")


class BiasDetector:
    """
    M1 Bias Detection.
    Primary: Logistic Regression on TF-IDF (WinoBias + StereoSet).
    Fallback: rule-based if .pkl not found.
    Same interface as original — detect(text, content_type) → dict.
    """

    # ── Supplementary rule layer (runs ALONGSIDE the ML model) ────────
    # These catch high-severity explicit slurs / patterns that TF-IDF
    # alone might miss when text is very short.
    EXPLICIT_BIAS_PATTERNS = [
        (r"\b(inferior|superior)\s+(race|gender|people|group)\b",   "explicit_slur",     0.95),
        (r"\b(subhuman|vermin|parasite)\b",                          "explicit_slur",     0.95),
        (r"\b(should be|deserve to be)\s+(eliminated|removed)\b",   "incitement",        0.90),
        (r"\b(hate|despise|loathe)\s+(all|every)\s+\w+\b",          "hate_language",     0.80),
        (r"\b(women|men)\s+(are|can'?t|cannot|should)\b",           "gender_stereotype", 0.70),
        (r"\bimmigrants?\s+(are|always|cause|bring)\b",             "ethnic_stereotype", 0.75),
    ]

    # Content-type bias on ML probability
    CONTENT_ADJUSTMENTS = {
        "academic": -0.05,   # academic text uses technical language → lower bias estimate
        "social":   +0.10,   # social media → slightly more weight on bias signals
        "news":      0.00,
        "text":      0.00,
    }

    def __init__(self):
        self._model      = None
        self._vectorizer = None
        self._ml_ready   = False
        self._load_models()
        self._compile_patterns()

    def _load_models(self):
        """Load trained pkl files produced by train_models.py."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self._vectorizer = pickle.load(f)
                self._ml_ready = True
                logger.info("[M1] Logistic Regression model loaded from models/")
                print("  [M1 Bias] ML model loaded ✓")
            except Exception as e:
                logger.warning(f"[M1] Model load failed: {e}. Using fallback.")
                print(f"  [M1 Bias] WARNING: model load failed ({e}) — using rule-based fallback")
        else:
            print("  [M1 Bias] WARNING: trained models not found — run train_models.py first. Using rule-based fallback.")

    def _compile_patterns(self):
        self._explicit_patterns = [
            (re.compile(p, re.IGNORECASE), label, w)
            for p, label, w in self.EXPLICIT_BIAS_PATTERNS
        ]

    # ──────────────────────────────────────────────────────────────────
    def detect(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Primary public method.  Same signature and return shape as before.
        """
        findings        = []
        word_count      = max(len(text.split()), 1)
        adjustment      = self.CONTENT_ADJUSTMENTS.get(content_type, 0.0)

        # ── 1.  ML inference (Logistic Regression) ────────────────────
        if self._ml_ready:
            raw_bias_prob = self._ml_predict(text)
            raw_bias_prob = max(0.0, min(1.0, raw_bias_prob + adjustment))
            score_ml      = max(0, round((1.0 - raw_bias_prob) * 100))
            findings.append({
                "type": "ml_classifier",
                "model": "Logistic Regression (WinoBias + StereoSet TF-IDF)",
                "bias_probability": round(raw_bias_prob, 4),
                "severity": raw_bias_prob
            })
        else:
            # Fallback: density-based score from explicit patterns only
            raw_bias_prob = 0.0
            score_ml      = 95   # optimistic default until overridden below

        # ── 2.  Explicit pattern layer (always runs) ──────────────────
        explicit_findings, explicit_severity = self._check_explicit(text)
        findings.extend(explicit_findings)

        if explicit_findings:
            density_factor = min(len(explicit_findings) / (word_count / 100), 1.0)
            pattern_bias   = (explicit_severity * 0.6) + (density_factor * 0.4)
            if self._ml_ready:
                # Blend: 70 % ML, 30 % pattern (but pattern can never improve score)
                blended = max(raw_bias_prob, 0.7 * raw_bias_prob + 0.3 * pattern_bias)
                score   = max(0, round((1.0 - blended) * 100))
            else:
                score = max(0, round((1.0 - pattern_bias) * 100))
        else:
            score = score_ml

        # ── 3.  Verdict ───────────────────────────────────────────────
        if score >= 80:
            verdict = "Neutral"
        elif score >= 50:
            verdict = "Mild Bias Detected"
        else:
            verdict = "Biased Content"

        return {
            "score":   score,
            "verdict": verdict,
            "findings": self._summary(findings, explicit_findings),
            "details": {
                "ml_used":           self._ml_ready,
                "model":             "Logistic Regression (WinoBias + StereoSet)" if self._ml_ready else "Rule-based fallback",
                "explicit_patterns": len(explicit_findings),
                "findings_list":     findings[:10],
            }
        }

    # ──────────────────────────────────────────────────────────────────
    def _ml_predict(self, text: str) -> float:
        """Return bias probability from trained Logistic Regression."""
        X    = self._vectorizer.transform([text])
        prob = self._model.predict_proba(X)[0]
        # Class 1 = biased
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def _check_explicit(self, text: str):
        findings  = []
        severities = []
        for pattern, label, weight in self._explicit_patterns:
            matches = pattern.findall(text)
            for m in matches:
                findings.append({
                    "type":     "explicit_pattern",
                    "category": label,
                    "match":    (m if isinstance(m, str) else " ".join(m))[:60],
                    "severity": weight,
                })
                severities.append(weight)
        avg_sev = sum(severities) / len(severities) if severities else 0.0
        return findings, avg_sev

    def _summary(self, findings: List[Dict], explicit: List[Dict]) -> str:
        if not findings:
            return "No significant bias indicators detected. Content appears neutral and balanced."
        parts = []
        if self._ml_ready:
            ml_f  = [f for f in findings if f["type"] == "ml_classifier"]
            if ml_f:
                prob = ml_f[0]["bias_probability"]
                parts.append(f"ML classifier bias probability: {prob:.0%}")
        if explicit:
            cats = set(f["category"] for f in explicit)
            parts.append(f"Explicit patterns: {', '.join(cats)}")
        return "; ".join(parts).capitalize() + "." if parts else "Bias indicators present."