"""
M3: Privacy Audit Module — ML VERSION
M3 has no ML classifier — train_models.py only produces m3_pii_config.json
(regex config validated on Enron Email Dataset).

This version loads that JSON config at runtime instead of using
hardcoded patterns, making it data-driven from train_models.py.
Same interface: audit(text) → dict.

Model files needed (run train_models.py first):
    backend/models/m3_pii_config.json
"""

import os
import re
import json
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Path ───────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_BASE_DIR, "..", "models", "m3_pii_config.json")


@dataclass
class PIIPattern:
    name:        str
    pattern:     str
    risk_level:  str
    risk_weight: float
    description: str


class PrivacyAuditor:
    """
    M3 Privacy Audit.
    Loads the 12-class PII regex config from m3_pii_config.json
    (produced and validated by train_models.py on Enron Email Dataset).
    Falls back to built-in patterns if config not found.
    Same interface: audit(text) → dict.
    """

    # ── Fallback patterns (identical to what train_models.py saves) ───
    _FALLBACK_PATTERNS = [
        PIIPattern("email",        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  "medium",   0.50, "Email address"),
        PIIPattern("phone_us",     r'\b(?:\+1[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}\b', "medium", 0.50, "US phone"),
        PIIPattern("phone_india",  r'(?:\+91[\s\-]?)?[6-9]\d{9}\b',                           "medium",   0.50, "Indian phone"),
        PIIPattern("phone_intl",   r'\+[1-9]\d{1,14}',                                        "medium",   0.50, "International phone"),
        PIIPattern("ssn",          r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',                        "critical", 0.95, "SSN"),
        PIIPattern("aadhaar",      r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b',                         "critical", 0.95, "Aadhaar"),
        PIIPattern("credit_card",  r'\b(?:\d{4}[-\s]?){3}\d{4}\b',                            "critical", 0.98, "Credit card"),
        PIIPattern("passport",     r'\b[A-Z]{1,2}\d{6,9}\b',                                  "high",     0.80, "Passport"),
        PIIPattern("ip_address",   r'\b(?:\d{1,3}\.){3}\d{1,3}\b',                           "low",      0.30, "IP address"),
        PIIPattern("date_of_birth",r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b', "medium", 0.60, "Date of birth"),
        PIIPattern("bank_account", r'\b\d{9,18}\b(?=.*(?:account|routing|bank))',              "critical", 0.95, "Bank account"),
        PIIPattern("medical_rec",  r'\b(?:MRN|MR|patient\.?id|medical\.?record|record\.?no)[\s:]*\d+\b', "high", 0.85, "Medical record"),
    ]

    _SENSITIVE_KEYWORDS = {
        "credentials": ["password", "passwd", "secret key", "api key", "auth token", "private key"],
        "financial":   ["bank account", "routing number", "cvv", "pin", "credit score"],
        "medical":     ["diagnosis", "patient", "prescription", "medical history", "blood type"],
        "personal":    ["social security", "mother's maiden name", "place of birth"],
    }

    _RISK_WEIGHTS = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}

    def __init__(self):
        self._config_loaded = False
        pii_patterns        = self._load_config()
        self._compiled      = [(p, re.compile(p.pattern, re.IGNORECASE)) for p in pii_patterns]

    # ── Config loading ─────────────────────────────────────────────────
    def _load_config(self) -> List[PIIPattern]:
        """
        Load m3_pii_config.json produced by train_models.py.
        Falls back to hardcoded patterns if file not found.
        """
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                patterns = []
                risk_w   = cfg.get("risk_weights", self._RISK_WEIGHTS)
                for name, info in cfg.get("patterns", {}).items():
                    patterns.append(PIIPattern(
                        name        = name,
                        pattern     = info["regex"],
                        risk_level  = info["risk_level"],
                        risk_weight = info["weight"],
                        description = name.replace("_", " ").title(),
                    ))
                self._config_loaded = True
                self._risk_weights  = risk_w
                print(f"  [M3 Privacy] Config loaded from models/m3_pii_config.json ({len(patterns)} patterns) ✓")
                return patterns
            except Exception as e:
                print(f"  [M3 Privacy] WARNING: config load failed ({e}) — using built-in patterns")

        print("  [M3 Privacy] WARNING: m3_pii_config.json not found — run train_models.py first. Using built-in patterns.")
        self._risk_weights = self._RISK_WEIGHTS
        return self._FALLBACK_PATTERNS

    # ── Primary method ─────────────────────────────────────────────────
    def audit(self, text: str) -> Dict[str, Any]:
        findings     = []
        risk_by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        # ── 1.  Regex PII scan ─────────────────────────────────────────
        for pii_pattern, compiled in self._compiled:
            for match in compiled.finditer(text):
                findings.append({
                    "type":         pii_pattern.name,
                    "description":  pii_pattern.description,
                    "value_masked": self._mask(match.group()),
                    "position":     match.start(),
                    "risk_level":   pii_pattern.risk_level,
                    "risk_weight":  pii_pattern.risk_weight,
                })
                risk_by_level[pii_pattern.risk_level] += 1

        # ── 2.  Sensitive keyword scan ────────────────────────────────
        tl = text.lower()
        kw_findings = []
        for category, keywords in self._SENSITIVE_KEYWORDS.items():
            for kw in keywords:
                if kw in tl:
                    kw_findings.append({
                        "type":        "sensitive_keyword",
                        "category":    category,
                        "keyword":     kw,
                        "risk_level":  "medium",
                        "risk_weight": 0.4,
                    })
        findings.extend(kw_findings)

        # ── 3.  Score (Enron-calibrated weighted risk formula) ─────────
        rw = self._risk_weights
        if findings:
            weighted_risk = (
                risk_by_level["critical"] * rw.get("critical", 1.0) +
                risk_by_level["high"]     * rw.get("high",     0.7) +
                risk_by_level["medium"]   * rw.get("medium",   0.4) +
                risk_by_level["low"]      * rw.get("low",      0.2) +
                len(kw_findings)          * 0.3
            )
            score = max(0, round((1 - min(weighted_risk / 5, 1.0)) * 100))
        else:
            score = 98

        # ── 4.  Verdict ───────────────────────────────────────────────
        if risk_by_level["critical"] > 0:
            verdict = "High Risk — Critical PII Detected"
        elif risk_by_level["high"] > 0:
            verdict = "PII Detected — Review Required"
        elif risk_by_level["medium"] > 0 or len(kw_findings) > 2:
            verdict = "PII Detected — Minor Concerns"
        else:
            verdict = "Safe — No PII Detected"

        return {
            "score":   score,
            "verdict": verdict,
            "findings": self._summary(findings, risk_by_level, kw_findings),
            "details": {
                "config_source":  "m3_pii_config.json (train_models.py)" if self._config_loaded else "built-in fallback",
                "pii_count":      len([f for f in findings if f["type"] != "sensitive_keyword"]),
                "keyword_count":  len(kw_findings),
                "risk_breakdown": risk_by_level,
                "findings_list":  findings[:15],
                "recommendations": self._recommendations(findings),
            }
        }

    # ── Helpers ────────────────────────────────────────────────────────
    def _mask(self, value: str) -> str:
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    def _summary(self, findings: List[Dict], risk_by_level: Dict, kw: List) -> str:
        if not findings:
            return "No PII detected. Content appears privacy-safe."
        parts = []
        total = sum(risk_by_level.values())
        for level in ("critical", "high", "medium"):
            if risk_by_level[level]:
                parts.append(f"{risk_by_level[level]} {level}-risk PII")
        if kw:
            parts.append(f"{len(kw)} sensitive keyword(s)")
        pii_types = set(f["type"] for f in findings if f["type"] != "sensitive_keyword")
        return f"Privacy audit found {total} PII instance(s): {'; '.join(parts)}. Types: {', '.join(pii_types)}." if parts else f"Detected {len(findings)} potential privacy concern(s)."

    def _recommendations(self, findings: List[Dict]) -> List[str]:
        recs  = []
        types = set(f["type"] for f in findings)
        if "credit_card" in types:
            recs.append("CRITICAL: Remove credit card numbers immediately.")
        if "ssn" in types or "aadhaar" in types:
            recs.append("CRITICAL: Government ID numbers — remove before sharing.")
        if "email" in types:
            recs.append("Consider masking email addresses.")
        if "phone_us" in types or "phone_india" in types:
            recs.append("Phone numbers detected — verify if disclosure is appropriate.")
        if any(f.get("category") == "credentials" for f in findings):
            recs.append("WARNING: Credential keywords — never share passwords or API keys.")
        if not recs and findings:
            recs.append("Review detected items before sharing content publicly.")
        return recs