"""
M4: Explainability Analysis Module — ML VERSION (FIXED)
Loads trained LinearSVC (CalibratedClassifierCV) + TF-IDF from train_models.py.

BUG FIXED: score_ml was set to None when _ml_ready=False, then used in arithmetic
(0.60 * None) causing TypeError. Now defaults to score_rule when models absent.

Model files needed (run train_models.py first):
    backend/models/m4_explain_model.pkl
    backend/models/m4_explain_vectorizer.pkl
"""

import os
import re
import pickle
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_BASE_DIR, "..", "models")

MODEL_PATH      = os.path.join(_MODEL_DIR, "m4_explain_model.pkl")
VECTORIZER_PATH = os.path.join(_MODEL_DIR, "m4_explain_vectorizer.pkl")


class ExplainabilityEngine:
    """
    M4 Explainability Analysis.
    Primary: LinearSVC (Calibrated) on TF-IDF (e-SNLI + CoS-E).
    Supplementary: 18-feature linguistic rule layer.
    Fallback: rule-based only if models not trained.
    Same interface: analyze(text) → dict.
    """

    REASONING_INDICATORS = [
        (r"\b(because|since|therefore|thus|hence|as a result|consequently)\b", "causal_reasoning",     0.15),
        (r"\b(first|second|third|finally|additionally|moreover|furthermore)\b",  "structured_reasoning", 0.10),
        (r"\b(for example|for instance|such as|e\.g\.|i\.e\.|to illustrate)\b",  "example_provision",    0.12),
        (r"\b(according to|based on|research shows|studies indicate|reported by)\b", "source_attribution",0.15),
        (r"\b(this means|in other words|to clarify|specifically|to put it simply)\b","clarification",     0.10),
    ]
    HEDGING_INDICATORS = [
        (r"\b(might|may|could|possibly|perhaps|likely|probably|potentially)\b", "uncertainty_hedge",  0.10),
        (r"\b(suggests?|appears?|seems?|indicates?|implies?)\b",                 "tentative_language", 0.08),
        (r"\b(in some cases|sometimes|often|generally|typically)\b",             "qualified_claim",    0.10),
        (r"\b(approximately|about|around|roughly|estimated)\b",                  "precision_qualifier",0.08),
    ]
    OPACITY_INDICATORS = [
        (r"\b(obviously|clearly|undeniably|unquestionably)\b",             "overconfidence",    -0.15),
        (r"\b(everyone knows|it's common knowledge|as we all know)\b",     "assumed_knowledge", -0.12),
        (r"\b(trust me|believe me|I guarantee)\b",                         "appeal_to_trust",   -0.20),
        (r"\b(always|never|all|none|every)\b",                             "absolute_claim",    -0.08),
    ]
    LOGIC_CONNECTORS = [
        "however", "although", "nevertheless", "on the other hand",
        "in contrast", "while", "whereas", "despite", "conversely",
        "even though", "yet", "that said",
    ]

    def __init__(self):
        self._model      = None
        self._vectorizer = None
        self._ml_ready   = False
        self._load_models()
        self._compile_patterns()

    def _load_models(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self._vectorizer = pickle.load(f)
                self._ml_ready = True
                print("  [M4 Explainability] ML model loaded ✓")
            except Exception as e:
                print(f"  [M4 Explainability] WARNING: model load failed ({e}) — rule-based fallback")
        else:
            print("  [M4 Explainability] WARNING: models not found — run train_models.py first. Rule-based fallback.")

    def _compile_patterns(self):
        self._reasoning = [(re.compile(p, re.IGNORECASE), t, w) for p, t, w in self.REASONING_INDICATORS]
        self._hedging   = [(re.compile(p, re.IGNORECASE), t, w) for p, t, w in self.HEDGING_INDICATORS]
        self._opacity   = [(re.compile(p, re.IGNORECASE), t, w) for p, t, w in self.OPACITY_INDICATORS]

    def analyze(self, text: str) -> Dict[str, Any]:
        tl         = text.lower()
        sentences  = re.split(r'[.!?]+', text)
        word_count = max(len(text.split()), 1)

        # findings is a dict for internal tracking
        findings = {"reasoning": [], "hedging": [], "opacity": [], "structure": {}}

        # ── 1. ML inference ───────────────────────────────────────────
        score_ml     = None
        ml_prob_high = None
        if self._ml_ready:
            try:
                ml_prob_high = self._ml_predict(text)
                score_ml     = max(0, round(ml_prob_high * 100))
                findings["ml_classifier"] = {
                    "model":        "LinearSVC (e-SNLI + CoS-E TF-IDF)",
                    "quality_prob": round(ml_prob_high, 4),
                }
            except Exception as e:
                print(f"  [M4] ML predict error: {e}")
                score_ml = None

        # ── 2. 18-feature linguistic rule layer ───────────────────────
        pos_score = 0.0
        neg_score = 0.0

        for pattern, itype, w in self._reasoning:
            ms = pattern.findall(tl)
            if ms:
                pos_score += w * min(len(ms), 5)
                findings["reasoning"].append({"type": itype, "count": len(ms)})

        for pattern, itype, w in self._hedging:
            ms = pattern.findall(tl)
            if ms:
                pos_score += w * min(len(ms), 4)
                findings["hedging"].append({"type": itype, "count": len(ms)})

        for pattern, itype, w in self._opacity:
            ms = pattern.findall(tl)
            if ms:
                neg_score += abs(w) * min(len(ms), 3)
                findings["opacity"].append({"type": itype, "count": len(ms)})

        # Structural features
        logic_count = sum(1 for c in self.LOGIC_CONNECTORS if c in tl)
        avg_sent    = word_count / max(len(sentences), 1)
        has_paras   = "\n\n" in text or text.count("\n") > 2
        struct_score = 0.0
        if logic_count >= 2:    struct_score += 0.15
        if 15 <= avg_sent <= 30: struct_score += 0.10
        if has_paras:            struct_score += 0.10
        if len(sentences) >= 3:  struct_score += 0.10
        pos_score += struct_score
        findings["structure"] = {
            "logic_connectors":   logic_count,
            "avg_sentence_length": round(avg_sent, 1),
            "has_paragraphs":     has_paras,
        }

        # Rule-based score
        raw        = pos_score - neg_score
        norm       = (raw + 0.5) / 2.0
        score_rule = max(0, min(100, round(norm * 100)))
        if word_count < 50:
            score_rule = min(score_rule, 70)

        # ── 3. FIXED: Blend — score_ml defaults to score_rule if None ─
        if score_ml is not None:
            score = round(0.60 * score_ml + 0.40 * score_rule)
        else:
            score = score_rule          # ← FIX: was 0.60 * None which crashed

        if word_count < 50:
            score = min(score, 70)

        score = max(0, min(100, score))

        # ── 4. Verdict ────────────────────────────────────────────────
        if score >= 75:
            verdict = "Explainable — Clear Reasoning"
        elif score >= 50:
            verdict = "Partially Explainable"
        elif score >= 30:
            verdict = "Opaque — Limited Transparency"
        else:
            verdict = "Inconsistent — Poor Explainability"

        return {
            "score":   score,
            "verdict": verdict,
            "findings": self._summary(findings, score),
            "details": {
                "ml_used":    self._ml_ready,
                "model":      "LinearSVC CalibratedClassifierCV (e-SNLI)" if self._ml_ready else "Rule-based fallback",
                "score_ml":   score_ml,
                "score_rule": score_rule,
                "pos_score":  round(pos_score, 3),
                "neg_score":  round(neg_score, 3),
                "structure":  findings["structure"],
            }
        }

    def _ml_predict(self, text: str) -> float:
        X    = self._vectorizer.transform([text])
        prob = self._model.predict_proba(X)[0]
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def _summary(self, findings: Dict, score: int) -> str:
        parts = []
        if "ml_classifier" in findings:
            qp = findings["ml_classifier"].get("quality_prob", 0)
            parts.append(f"ML quality probability: {qp:.0%}")
        r = len(findings.get("reasoning", []))
        h = len(findings.get("hedging", []))
        o = len(findings.get("opacity", []))
        if r:
            parts.append(f"{r} reasoning indicator(s)")
        if h:
            parts.append("appropriate uncertainty language")
        if o:
            parts.append(f"{o} opacity concern(s)")
        lc = findings.get("structure", {}).get("logic_connectors", 0)
        if lc >= 2:
            parts.append("logical connectors detected")
        if not parts:
            return ("Content shows reasonable transparency." if score >= 60
                    else "Limited explainability. Add reasoning chains and source attribution.")
        return "; ".join(parts).capitalize() + "."