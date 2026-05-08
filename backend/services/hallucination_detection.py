"""
M2: Hallucination Detection Module

Detects factual inaccuracies and unverifiable claims using:
- Wikipedia API fact verification
- Known fact patterns
- Claim extraction and validation
- Confidence estimation
"""

import re
import json
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Tuple
from datetime import datetime


class HallucinationDetector:
    """
    Hybrid hallucination detection combining:
    - External fact verification (Wikipedia)
    - Known fact database checking
    - Claim pattern analysis
    - Confidence scoring
    """
    
    # Known facts for quick verification
    KNOWN_FACTS = {
        "eiffel tower": {
            "year_built": 1889,
            "architect": "gustave eiffel",
            "location": "paris",
            "height_meters": 330,
            "original_purpose": "1889 world's fair"
        },
        "metformin": {
            "use": "type 2 diabetes",
            "type": "medication",
            "class": "biguanide"
        },
        "cold fusion": {
            "status": "not scientifically validated",
            "controversy": "pons and fleischmann 1989",
            "consensus": "not reproducible"
        }
    }
    
    # Hallucination red flags
    HALLUCINATION_PATTERNS = [
        # Overly precise statistics
        (r"exactly\s+\d{4,}", "suspiciously precise number", 0.4),
        (r"\d{2,3}\.\d{2,}%", "overly precise percentage", 0.3),
        # Unverifiable study references
        (r"(according to|based on)\s+a\s+\d{4}\s+study", "unverifiable study citation", 0.5),
        (r"(harvard|stanford|mit|oxford)\s+(research|study|scientists)\s+(shows?|proves?|confirms?)", "appeal to authority without citation", 0.4),
        # Absolute claims
        (r"(solves?|eliminates?|cures?)\s+(the|all)\s+\w+\s+(problem|crisis|disease)", "absolute solution claim", 0.5),
        (r"(100%|completely|totally|perfectly)\s+(effective|accurate|safe)", "absolute effectiveness claim", 0.6),
        # Future predictions as fact
        (r"will\s+(definitely|certainly|absolutely)\s+\w+", "overconfident prediction", 0.3),
        (r"is guaranteed to", "guarantee claim", 0.4),
        # Made-up statistics
        (r"\d{2,3}%\s+of\s+(all\s+)?(people|patients|users|cases)", "unverifiable statistic", 0.4),
        # Temporal inconsistencies
        (r"(in\s+)?(1\d{3}|20[3-9]\d)\s+.{0,30}\s+(discovered|invented|published)", "potential temporal error", 0.3),
    ]
    
    # Wikipedia API
    WIKIPEDIA_API = "[en.wikipedia.org](https://en.wikipedia.org/w/api.php)"
    
    def __init__(self, enable_api: bool = True):
        self.enable_api = enable_api
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns."""
        self.compiled_patterns = [
            (re.compile(p, re.IGNORECASE), desc, weight) 
            for p, desc, weight in self.HALLUCINATION_PATTERNS
        ]
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the text.
        
        Args:
            text: Content to verify
        
        Returns:
            Dictionary with score, verdict, and findings
        """
        findings = []
        verification_results = []
        
        # 1. Pattern-based detection
        pattern_score, pattern_findings = self._check_patterns(text)
        findings.extend(pattern_findings)
        
        # 2. Extract and verify claims
        claims = self._extract_claims(text)
        for claim in claims[:5]:  # Limit API calls
            verification = self._verify_claim(claim)
            verification_results.append(verification)
            if verification["status"] == "contradicted":
                findings.append({
                    "type": "factual_error",
                    "claim": claim[:100],
                    "evidence": verification.get("evidence", ""),
                    "severity": 0.8
                })
            elif verification["status"] == "unverifiable":
                findings.append({
                    "type": "unverifiable_claim",
                    "claim": claim[:100],
                    "severity": 0.4
                })
        
        # 3. Check against known facts
        known_fact_issues = self._check_known_facts(text)
        findings.extend(known_fact_issues)
        
        # 4. Calculate final score
        if findings:
            # Weight by severity
            total_severity = sum(f.get("severity", 0.5) for f in findings)
            avg_severity = total_severity / len(findings)
            
            # Factor in number of issues relative to text length
            sentences = len(re.split(r'[.!?]+', text))
            density = min(len(findings) / max(sentences, 1), 1.0)
            
            raw_hallucination = (avg_severity * 0.7 + density * 0.3)
            score = max(0, round((1 - raw_hallucination) * 100))
        else:
            score = 90  # Good score if no issues detected
        
        # Adjust based on verification results
        verified_count = sum(1 for v in verification_results if v["status"] == "verified")
        if verification_results:
            verification_ratio = verified_count / len(verification_results)
            score = round(score * 0.7 + verification_ratio * 100 * 0.3)
        
        # Determine verdict
        if score >= 75:
            verdict = "Factually Sound"
        elif score >= 50:
            verdict = "Uncertain — Verification Recommended"
        else:
            verdict = "Hallucination Risk — Contains Unverified/False Claims"
        
        findings_summary = self._generate_summary(findings, verification_results)
        
        return {
            "score": score,
            "verdict": verdict,
            "findings": findings_summary,
            "details": {
                "pattern_findings": len([f for f in findings if f["type"] != "factual_error"]),
                "factual_errors": len([f for f in findings if f["type"] == "factual_error"]),
                "claims_verified": verified_count,
                "claims_checked": len(verification_results),
                "findings_list": findings[:10]
            }
        }
    
    def _check_patterns(self, text: str) -> Tuple[float, List[Dict]]:
        """Check for hallucination patterns."""
        findings = []
        total_weight = 0.0
        
        for pattern, description, weight in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                match_text = match if isinstance(match, str) else str(match)
                findings.append({
                    "type": "suspicious_pattern",
                    "pattern": description,
                    "match": match_text[:50],
                    "severity": weight
                })
                total_weight += weight
        
        return total_weight, findings
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Look for factual claim indicators
            claim_indicators = [
                r'\b(is|are|was|were|has|have|had)\b',
                r'\b(states?|claims?|shows?|proves?|confirms?)\b',
                r'\b(percent|%|\d+\s+(million|billion|thousand))\b',
                r'\b(in\s+\d{4}|since\s+\d{4}|by\s+\d{4})\b',
            ]
            
            for indicator in claim_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break
        
        return claims[:10]  # Limit number of claims
    
    def _verify_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a claim using Wikipedia API."""
        if not self.enable_api:
            return {"status": "unchecked", "claim": claim}
        
        # Extract key entities from claim
        entities = self._extract_entities(claim)
        
        if not entities:
            return {"status": "unverifiable", "claim": claim, "reason": "no entities found"}
        
        # Search Wikipedia for verification
        for entity in entities[:2]:
            wiki_result = self._search_wikipedia(entity)
            
            if wiki_result.get("found"):
                snippet = wiki_result.get("snippet", "").lower()
                claim_lower = claim.lower()
                
                # Simple entailment check - look for contradictions
                if self._check_contradiction(claim_lower, snippet):
                    return {
                        "status": "contradicted",
                        "claim": claim,
                        "entity": entity,
                        "evidence": snippet[:200]
                    }
                
                # Check for support
                entity_words = set(entity.lower().split())
                snippet_words = set(snippet.split())
                overlap = len(entity_words.intersection(snippet_words))
                
                if overlap >= len(entity_words) * 0.5:
                    return {
                        "status": "verified",
                        "claim": claim,
                        "entity": entity,
                        "source": "wikipedia"
                    }
        
        return {"status": "unverifiable", "claim": claim}
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        # Capitalized phrases (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns)
        
        # Numbers with context
        numbers_with_context = re.findall(r'(\d+(?:\.\d+)?)\s*(percent|%|meters?|feet|years?|million|billion)', text, re.IGNORECASE)
        
        return list(set(entities))[:5]
    
    def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for a term."""
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            }
            
            url = f"{self.WIKIPEDIA_API}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
            
            search_results = data.get("query", {}).get("search", [])
            
            if search_results:
                # Clean HTML tags from snippet
                snippet = search_results[0].get("snippet", "")
                snippet = re.sub(r'<[^>]+>', '', snippet)
                
                return {
                    "found": True,
                    "title": search_results[0].get("title"),
                    "snippet": snippet
                }
            
            return {"found": False}
            
        except Exception as e:
            return {"found": False, "error": str(e)}
    
    def _check_contradiction(self, claim: str, evidence: str) -> bool:
        """Simple contradiction detection."""
        # Check for year contradictions
        claim_years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', claim)
        evidence_years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', evidence)
        
        if claim_years and evidence_years:
            # If different years mentioned for same event
            claim_year = int(claim_years[0])
            for ev_year in evidence_years:
                if abs(int(ev_year) - claim_year) > 5:
                    return True
        
        # Check for numerical contradictions
        claim_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(meters?|feet|tall|high)', claim)
        evidence_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(meters?|feet|tall|high)', evidence)
        
        # More sophisticated checks could be added here
        
        return False
    
    def _check_known_facts(self, text: str) -> List[Dict]:
        """Check text against known facts database."""
        issues = []
        text_lower = text.lower()
        
        for topic, facts in self.KNOWN_FACTS.items():
            if topic in text_lower:
                # Check for contradictions with known facts
                if "year_built" in facts:
                    year_pattern = re.search(rf'{topic}.*?(1[89]\d{{2}}|20\d{{2}})', text_lower)
                    if year_pattern:
                        mentioned_year = int(year_pattern.group(1))
                        if mentioned_year != facts["year_built"]:
                            issues.append({
                                "type": "factual_error",
                                "category": "incorrect_date",
                                "claim": f"Stated {topic} built in {mentioned_year}",
                                "fact": f"Actually built in {facts['year_built']}",
                                "severity": 0.9
                            })
                
                if "location" in facts:
                    # Check if wrong location mentioned
                    if facts["location"] not in text_lower:
                        wrong_locations = ["london", "new york", "berlin", "tokyo", "rome"]
                        for wrong in wrong_locations:
                            if wrong in text_lower and f"{topic}" in text_lower:
                                issues.append({
                                    "type": "factual_error",
                                    "category": "incorrect_location",
                                    "claim": f"Placed {topic} in {wrong}",
                                    "fact": f"Actually located in {facts['location']}",
                                    "severity": 0.85
                                })
                                break
        
        return issues
    
    def _generate_summary(self, findings: List[Dict], verifications: List[Dict]) -> str:
        """Generate human-readable summary."""
        if not findings and all(v["status"] == "verified" for v in verifications):
            return "Content appears factually sound. Key claims verified against reference sources."
        
        parts = []
        
        factual_errors = [f for f in findings if f["type"] == "factual_error"]
        if factual_errors:
            parts.append(f"{len(factual_errors)} factual error(s) detected")
        
        suspicious = [f for f in findings if f["type"] == "suspicious_pattern"]
        if suspicious:
            parts.append(f"{len(suspicious)} suspicious claim pattern(s)")
        
        unverifiable = [v for v in verifications if v["status"] == "unverifiable"]
        if unverifiable:
            parts.append(f"{len(unverifiable)} unverifiable claim(s)")
        
        if parts:
            return f"Hallucination risk identified: {'; '.join(parts)}. Recommend fact-checking before use."
        
        return "Some claims could not be fully verified. Exercise caution with factual assertions."
