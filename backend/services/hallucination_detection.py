"""
M2: Hallucination Detection Module — ML VERSION (FIXED)
Loads trained Random Forest + TF-IDF from train_models.py output.

BUG FIXED: self._kb.items() now has type-safety guard so if
data/knowledge_base.json is a list (or any non-dict), it falls back
to the built-in KNOWN_FACTS dict instead of crashing.

Model files needed (run train_models.py first):
    backend/models/m2_hallucination_model.pkl
    backend/models/m2_hallucination_vectorizer.pkl
"""

import os
import re
import json
import pickle
import logging
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "..", "models")

MODEL_PATH      = os.path.join(_MODEL_DIR, "m2_hallucination_model.pkl")
VECTORIZER_PATH = os.path.join(_MODEL_DIR, "m2_hallucination_vectorizer.pkl")
KB_PATHS        = [
    os.path.join(_BASE_DIR, "..", "data", "knowledge_base.json"),
    "data/knowledge_base.json",
    "../data/knowledge_base.json",
]

# ── Built-in fallback knowledge base ──────────────────────────────────
_BUILTIN_KB = {
    "eiffel tower": {
        "year_built": 1889,
        "architect": "gustave eiffel",
        "location": "paris",
        "height_meters": 330,
    },
    "metformin": {
        "use": "type 2 diabetes",
        "class": "biguanide",
    },
    "cold fusion": {
        "status": "not scientifically validated",
    },
}

# ── 10 FEVER-derived patterns ──────────────────────────────────────────
_FEVER_PATTERNS = [
    (r"exactly\s+\d{4,}",
     "suspiciously_precise_number", 0.4),
    (r"\d{2,3}\.\d{2,}%",
     "overly_precise_percentage", 0.3),
    (r"(according to|based on)\s+a\s+\d{4}\s+study",
     "unverifiable_study_citation", 0.5),
    (r"(harvard|stanford|mit|oxford)\s+(research|study)\s+(proves?|shows?|confirms?)",
     "appeal_to_authority", 0.4),
    (r"(solves?|eliminates?|cures?)\s+(the|all)\s+\w+\s+(problem|crisis|disease)",
     "absolute_solution", 0.5),
    (r"(100%|completely|totally|perfectly)\s+(effective|accurate|safe)",
     "absolute_effectiveness", 0.6),
    (r"will\s+(definitely|certainly|absolutely)\s+\w+",
     "overconfident_prediction", 0.3),
    (r"is guaranteed to",
     "guarantee_claim", 0.4),
    (r"\d{2,3}%\s+of\s+(all\s+)?(people|patients|users|cases)",
     "unverifiable_statistic", 0.4),
    (r"in\s+(1[89]\d{2}|20[3-9]\d)\s+.{0,30}\s+(discovered|invented|published)",
     "temporal_error", 0.3),
]


class HallucinationDetector:
    """
    M2 Hallucination Detection.
    Primary: Random Forest on TF-IDF (FEVER dataset).
    Supplementary: Wikipedia RAV + knowledge-base fact checking.
    Fallback: FEVER regex patterns only if models not trained.
    """

    WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, enable_api: bool = True):
        self.enable_api  = enable_api
        self._model      = None
        self._vectorizer = None
        self._ml_ready   = False
        self._kb         = self._load_kb()      # always a dict after _load_kb
        self._load_models()
        self._compile_patterns()

    # ── Model loading ──────────────────────────────────────────────────
    def _load_models(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self._vectorizer = pickle.load(f)
                self._ml_ready = True
                print("  [M2 Hallucination] ML model loaded ✓")
            except Exception as e:
                print(f"  [M2 Hallucination] WARNING: model load failed ({e}) — rule-based fallback")
        else:
            print("  [M2 Hallucination] WARNING: models not found — run train_models.py first. Rule-based fallback.")

    def _load_kb(self) -> dict:
        """
        Load knowledge_base.json.
        FIXED: validates that the loaded data is a dict.
        If it's a list or any other type, falls back to _BUILTIN_KB.
        """
        for path in KB_PATHS:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # ── TYPE SAFETY GUARD ──────────────────────────────
                    if isinstance(data, dict):
                        return data
                    else:
                        print(f"  [M2] WARNING: knowledge_base.json is {type(data).__name__}, not dict. Using built-in KB.")
                        return _BUILTIN_KB
                except Exception as e:
                    print(f"  [M2] KB load error: {e}")
        return _BUILTIN_KB

    def _compile_patterns(self):
        self._patterns = [
            (re.compile(p, re.IGNORECASE), desc, w)
            for p, desc, w in _FEVER_PATTERNS
        ]

    # ── Primary method ─────────────────────────────────────────────────
    def detect(self, text: str) -> Dict[str, Any]:
        findings             = []
        verification_results = []

        # ── 1. ML inference ───────────────────────────────────────────
        ml_prob = None
        if self._ml_ready:
            try:
                ml_prob = self._ml_predict(text)
                findings.append({
                    "type":  "ml_classifier",
                    "model": "Random Forest (FEVER TF-IDF)",
                    "hallucination_probability": round(ml_prob, 4),
                    "severity": ml_prob,
                })
            except Exception as e:
                print(f"  [M2] ML predict error: {e}")
                ml_prob = None

        # ── 2. FEVER pattern layer (always runs) ──────────────────────
        for pattern, desc, weight in self._patterns:
            for m in pattern.findall(text):
                findings.append({
                    "type":     "suspicious_pattern",
                    "pattern":  desc,
                    "match":    (m if isinstance(m, str) else str(m))[:60],
                    "severity": weight,
                })

        # ── 3. Knowledge-base fact check ──────────────────────────────
        # _kb is guaranteed to be a dict (validated in _load_kb)
        findings.extend(self._check_known_facts(text))

        # ── 4. Wikipedia RAV ──────────────────────────────────────────
        if self.enable_api:
            for claim in self._extract_claims(text)[:4]:
                try:
                    vr = self._verify_claim(claim)
                    verification_results.append(vr)
                    if vr["status"] == "contradicted":
                        findings.append({
                            "type":     "factual_error",
                            "claim":    claim[:100],
                            "evidence": vr.get("evidence", ""),
                            "severity": 0.8,
                        })
                except Exception:
                    pass

        # ── 5. Score ──────────────────────────────────────────────────
        if ml_prob is not None:
            score = max(0, round((1.0 - ml_prob) * 100))
            pattern_findings = [f for f in findings if f.get("type") == "suspicious_pattern"]
            if pattern_findings:
                avg_pat = sum(f["severity"] for f in pattern_findings) / len(pattern_findings)
                blended = max(ml_prob, 0.6 * ml_prob + 0.4 * avg_pat)
                score   = max(0, round((1.0 - blended) * 100))
        elif findings:
            sev_list  = [f.get("severity", 0.4) for f in findings if "severity" in f]
            avg_sev   = sum(sev_list) / len(sev_list) if sev_list else 0.4
            sentences = max(len(re.split(r'[.!?]+', text)), 1)
            density   = min(len(findings) / sentences, 1.0)
            score     = max(0, round((1.0 - (avg_sev * 0.7 + density * 0.3)) * 100))
        else:
            score = 90

        if verification_results:
            verified = sum(1 for v in verification_results if v.get("status") == "verified")
            vr_ratio = verified / len(verification_results)
            score    = round(score * 0.7 + vr_ratio * 100 * 0.3)

        score = max(0, min(100, score))

        # ── 6. Verdict ────────────────────────────────────────────────
        if score >= 75:
            verdict = "Factually Sound"
        elif score >= 50:
            verdict = "Uncertain — Verification Recommended"
        else:
            verdict = "Hallucination Risk — Contains Unverified/False Claims"

        return {
            "score":   score,
            "verdict": verdict,
            "findings": self._summary(findings, verification_results),
            "details": {
                "ml_used":         self._ml_ready,
                "model":           "Random Forest (FEVER TF-IDF)" if self._ml_ready else "Rule-based fallback",
                "factual_errors":  len([f for f in findings if f.get("type") == "factual_error"]),
                "pattern_flags":   len([f for f in findings if f.get("type") == "suspicious_pattern"]),
                "claims_verified": sum(1 for v in verification_results if v.get("status") == "verified"),
                "claims_checked":  len(verification_results),
                "findings_list":   findings[:10],
            }
        }

    def _ml_predict(self, text: str) -> float:
        X    = self._vectorizer.transform([text])
        prob = self._model.predict_proba(X)[0]
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def _check_known_facts(self, text: str) -> List[Dict]:
        """self._kb is always a dict here (validated in _load_kb)."""
        issues = []
        tl = text.lower()
        for topic, facts in self._kb.items():          # safe — _kb is a dict
            if not isinstance(facts, dict):
                continue                                 # skip malformed entries
            if topic not in tl:
                continue
            if "year_built" in facts:
                m = re.search(
                    rf"{re.escape(topic)}.{{0,60}}(1[89]\d{{2}}|20\d{{2}})", tl)
                if m:
                    yr = int(m.group(1))
                    if yr != facts["year_built"]:
                        issues.append({
                            "type": "factual_error", "category": "incorrect_date",
                            "claim": f"Stated {topic} built in {yr}",
                            "fact":  f"Actually built in {facts['year_built']}",
                            "severity": 0.9,
                        })
            if "location" in facts and facts["location"] not in tl:
                for wrong in ["london", "new york", "berlin", "tokyo", "rome"]:
                    if wrong in tl:
                        issues.append({
                            "type": "factual_error", "category": "incorrect_location",
                            "claim": f"Placed {topic} in {wrong}",
                            "fact":  f"Actually in {facts['location']}",
                            "severity": 0.85,
                        })
                        break
        return issues

    def _extract_claims(self, text: str) -> List[str]:
        sentences  = re.split(r'[.!?]+', text)
        indicators = [r'\b(is|are|was|were)\b', r'\b\d{4}\b',
                      r'\b\d+%\b', r'\baccording to\b']
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            if any(re.search(p, s, re.IGNORECASE) for p in indicators):
                claims.append(s)
        return claims[:6]

    def _verify_claim(self, claim: str) -> Dict[str, Any]:
        if not self.enable_api:
            return {"status": "unchecked", "claim": claim}
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        for entity in list(set(entities))[:2]:
            result = self._search_wikipedia(entity)
            if result.get("found"):
                snippet = result.get("snippet", "").lower()
                if self._check_contradiction(claim.lower(), snippet):
                    return {"status": "contradicted", "claim": claim,
                            "entity": entity, "evidence": snippet[:200]}
                if any(w in snippet for w in entity.lower().split()):
                    return {"status": "verified", "claim": claim, "entity": entity}
        return {"status": "unverifiable", "claim": claim}

    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        try:
            params = {"action": "query", "list": "search",
                      "srsearch": query, "format": "json", "srlimit": 1}
            url = f"{self.WIKIPEDIA_API}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "AICES/3.0"})
            with urllib.request.urlopen(req, timeout=4) as r:
                data = json.loads(r.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                snippet = re.sub(r'<[^>]+>', '', results[0].get("snippet", ""))
                return {"found": True, "snippet": snippet}
        except Exception:
            pass
        return {"found": False}

    def _check_contradiction(self, claim: str, evidence: str) -> bool:
        cy = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', claim)
        ey = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', evidence)
        if cy and ey:
            return any(abs(int(c) - int(e)) > 5 for c in cy for e in ey)
        return False

    def _summary(self, findings: List[Dict], verifications: List[Dict]) -> str:
        if not findings and all(v.get("status") == "verified" for v in verifications if verifications):
            return "Content appears factually sound. Key claims verified."
        parts = []
        ml_f = [f for f in findings if f.get("type") == "ml_classifier"]
        if ml_f:
            prob = ml_f[0].get("hallucination_probability", 0)
            parts.append(f"ML hallucination probability: {prob:.0%}")
        fe = [f for f in findings if f.get("type") == "factual_error"]
        sp = [f for f in findings if f.get("type") == "suspicious_pattern"]
        if fe:
            parts.append(f"{len(fe)} factual error(s)")
        if sp:
            parts.append(f"{len(sp)} suspicious pattern(s)")
        uv = [v for v in verifications if v.get("status") == "unverifiable"]
        if uv:
            parts.append(f"{len(uv)} unverifiable claim(s)")
        return ("; ".join(parts).capitalize() + ".") if parts else "Some claims could not be fully verified."