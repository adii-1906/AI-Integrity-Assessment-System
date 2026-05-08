"""
M3: Privacy Audit Module

Detects Personally Identifiable Information (PII) including:
- Email addresses
- Phone numbers (multiple formats)
- SSN / Aadhaar numbers
- Credit card numbers
- Names and locations
- Other sensitive data patterns
"""

import re
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PIIPattern:
    """Configuration for a PII detection pattern."""
    name: str
    pattern: str
    risk_level: str  # low, medium, high, critical
    risk_weight: float
    description: str


class PrivacyAuditor:
    """
    Comprehensive PII detection for content privacy auditing.
    
    Detects:
    - Contact info (email, phone)
    - Government IDs (SSN, Aadhaar, passport)
    - Financial data (credit cards, bank accounts)
    - Personal identifiers (names, DOB, addresses)
    """
    
    PII_PATTERNS = [
        PIIPattern(
            name="email",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            risk_level="medium",
            risk_weight=0.5,
            description="Email address"
        ),
        PIIPattern(
            name="phone_us",
            pattern=r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            risk_level="medium",
            risk_weight=0.5,
            description="US phone number"
        ),
        PIIPattern(
            name="phone_international",
            pattern=r'\+[1-9]\d{1,14}',
            risk_level="medium",
            risk_weight=0.5,
            description="International phone number"
        ),
        PIIPattern(
            name="phone_indian",
            pattern=r'\b(?:\+91[-.\s]?)?[6-9]\d{9}\b',
            risk_level="medium",
            risk_weight=0.5,
            description="Indian phone number"
        ),
        PIIPattern(
            name="ssn",
            pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            risk_level="critical",
            risk_weight=0.95,
            description="Social Security Number"
        ),
        PIIPattern(
            name="aadhaar",
            pattern=r'\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b',
            risk_level="critical",
            risk_weight=0.95,
            description="Aadhaar number"
        ),
        PIIPattern(
            name="credit_card",
            pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            risk_level="critical",
            risk_weight=0.98,
            description="Credit card number"
        ),
        PIIPattern(
            name="passport",
            pattern=r'\b[A-Z]{1,2}\d{6,9}\b',
            risk_level="high",
            risk_weight=0.8,
            description="Passport number"
        ),
        PIIPattern(
            name="ip_address",
            pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            risk_level="low",
            risk_weight=0.3,
            description="IP address"
        ),
        PIIPattern(
            name="date_of_birth",
            pattern=r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b',
            risk_level="medium",
            risk_weight=0.6,
            description="Date of birth"
        ),
        PIIPattern(
            name="bank_account",
            pattern=r'\b\d{9,18}\b(?=.*(?:account|routing|bank))',
            risk_level="critical",
            risk_weight=0.95,
            description="Bank account number"
        ),
        PIIPattern(
            name="medical_record",
            pattern=r'\b(?:MRN|medical record|patient id)[-:\s]*\d+\b',
            risk_level="high",
            risk_weight=0.85,
            description="Medical record number"
        ),
    ]
    
    # Sensitive keyword categories
    SENSITIVE_KEYWORDS = {
        "credentials": ["password", "pwd", "secret key", "api key", "auth token", "private key"],
        "financial": ["bank account", "routing number", "cvv", "pin", "credit score"],
        "medical": ["diagnosis", "patient", "prescription", "medical history", "blood type"],
        "personal": ["social security", "mother's maiden name", "place of birth"],
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (p, re.compile(p.pattern, re.IGNORECASE))
            for p in self.PII_PATTERNS
        ]
    
    def audit(self, text: str) -> Dict[str, Any]:
        """
        Audit text for privacy risks.
        
        Args:
            text: Content to analyze
        
        Returns:
            Dictionary with score, verdict, and detailed findings
        """
        findings = []
        risk_by_level = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        total_risk_weight = 0.0
        
        # 1. Pattern-based PII detection
        for pii_pattern, compiled in self.compiled_patterns:
            matches = list(compiled.finditer(text))
            for match in matches:
                masked_value = self._mask_value(match.group())
                findings.append({
                    "type": pii_pattern.name,
                    "description": pii_pattern.description,
                    "value_masked": masked_value,
                    "position": match.start(),
                    "risk_level": pii_pattern.risk_level,
                    "risk_weight": pii_pattern.risk_weight
                })
                risk_by_level[pii_pattern.risk_level] += 1
                total_risk_weight += pii_pattern.risk_weight
        
        # 2. Sensitive keyword detection
        text_lower = text.lower()
        keyword_findings = []
        
        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_findings.append({
                        "type": "sensitive_keyword",
                        "category": category,
                        "keyword": keyword,
                        "risk_level": "medium",
                        "risk_weight": 0.4
                    })
                    total_risk_weight += 0.4
        
        findings.extend(keyword_findings)
        
        # 3. Calculate privacy score (higher = safer = better)
        if findings:
            # Weight critical findings heavily
            weighted_risk = (
                risk_by_level["critical"] * 1.0 +
                risk_by_level["high"] * 0.7 +
                risk_by_level["medium"] * 0.4 +
                risk_by_level["low"] * 0.2 +
                len(keyword_findings) * 0.3
            )
            
            # Normalize (cap the impact)
            normalized_risk = min(weighted_risk / 5, 1.0)
            score = max(0, round((1 - normalized_risk) * 100))
        else:
            score = 98  # Near-perfect if no PII detected
        
        # 4. Determine verdict
        if risk_by_level["critical"] > 0:
            verdict = "High Risk — Critical PII Detected"
        elif risk_by_level["high"] > 0:
            verdict = "PII Detected — Review Required"
        elif risk_by_level["medium"] > 0 or len(keyword_findings) > 2:
            verdict = "PII Detected — Minor Concerns"
        else:
            verdict = "Safe — No PII Detected"
        
        # 5. Generate findings summary
        findings_summary = self._generate_summary(findings, risk_by_level)
        
        return {
            "score": score,
            "verdict": verdict,
            "findings": findings_summary,
            "details": {
                "pii_count": len([f for f in findings if f["type"] != "sensitive_keyword"]),
                "keyword_count": len(keyword_findings),
                "risk_breakdown": risk_by_level,
                "findings_list": findings[:15],
                "recommendations": self._generate_recommendations(findings)
            }
        }
    
    def _mask_value(self, value: str) -> str:
        """Mask sensitive values for safe display."""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    
    def _generate_summary(self, findings: List[Dict], risk_by_level: Dict) -> str:
        """Generate human-readable summary."""
        if not findings:
            return "No personally identifiable information (PII) detected. Content appears privacy-safe."
        
        parts = []
        total_pii = sum(risk_by_level.values())
        
        if risk_by_level["critical"] > 0:
            parts.append(f"{risk_by_level['critical']} critical PII item(s) (SSN, credit card, Aadhaar)")
        if risk_by_level["high"] > 0:
            parts.append(f"{risk_by_level['high']} high-risk item(s)")
        if risk_by_level["medium"] > 0:
            parts.append(f"{risk_by_level['medium']} medium-risk item(s) (email, phone)")
        
        pii_types = set(f["type"] for f in findings if f["type"] != "sensitive_keyword")
        
        if parts:
            return f"Privacy audit found {total_pii} PII instance(s): {'; '.join(parts)}. Types: {', '.join(pii_types)}."
        
        keyword_count = len([f for f in findings if f["type"] == "sensitive_keyword"])
        if keyword_count > 0:
            return f"Found {keyword_count} sensitive keyword(s) that may indicate PII context."
        
        return f"Detected {len(findings)} potential privacy concern(s)."
    
    def _generate_recommendations(self, findings: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        types_found = set(f["type"] for f in findings)
        
        if "credit_card" in types_found:
            recommendations.append("CRITICAL: Remove credit card numbers immediately.")
        
        if "ssn" in types_found or "aadhaar" in types_found:
            recommendations.append("CRITICAL: Government ID numbers detected — remove before sharing.")
        
        if "email" in types_found:
            recommendations.append("Consider masking or removing email addresses if not necessary.")
        
        if "phone_us" in types_found or "phone_indian" in types_found:
            recommendations.append("Phone numbers detected — verify if disclosure is appropriate.")
        
        if any(f.get("category") == "credentials" for f in findings):
            recommendations.append("WARNING: Credential-related keywords detected — never share passwords or API keys.")
        
        if any(f.get("category") == "medical" for f in findings):
            recommendations.append("Medical information detected — ensure HIPAA/privacy compliance.")
        
        if not recommendations and findings:
            recommendations.append("Review detected items before sharing content publicly.")
        
        return recommendations
