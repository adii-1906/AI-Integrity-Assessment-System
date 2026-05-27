"""
M1: Bias Detection Module
DATASET VERSION - No API calls
Uses WinoBias (GitHub) + StereoSet (HuggingFace) derived lexicons.
Same class name, same method signature, same return structure as original.
"""

import re
from typing import Dict, Any, List
from collections import defaultdict


class BiasDetector:
    """
    Multi-method bias detection using WinoBias + StereoSet datasets.
    Replaces original - same interface, no API calls.
    """

    # ── Lexicons derived from WinoBias (github.com/uclanlp/corefBias) ──
    DEMOGRAPHIC_BIAS = {
        "gender": {
            "keywords": [
                "mankind", "manmade", "man-made", "chairman", "fireman",
                "policeman", "stewardess", "housewife", "career woman",
                "male nurse", "lady doctor", "bossy", "hysterical", "shrill",
                "emotional nature", "too emotional", "women are too",
                "women tend to be", "men are naturally", "men are better",
                "female engineer", "girl boss", "not suited for leadership",
                "woman's place", "man's job", "acts like a man",
                "emotional woman", "emotional female", "irrational woman",
            ],
            "stereotype_patterns": [
                r"women\s+(are|tend to be|should)\s+\w+",
                r"men\s+(are|tend to be|should)\s+\w+",
                r"(he|his|him)\s+.{0,30}\s+(doctor|engineer|ceo|boss|leader)",
                r"(she|her)\s+.{0,30}\s+(nurse|secretary|teacher|caregiver)",
                r"women?\s+(?:always|never|can'?t|cannot)\s+\w+",
                r"men?\s+(?:always|never|can'?t|cannot)\s+\w+",
            ],
            "weight": 0.8
        },
        "racial": {
            "keywords": [
                "exotic", "articulate", "urban", "ghetto", "thug",
                "illegal alien", "colored", "oriental", "ethnic",
                "foreign-looking", "inner city", "those people",
                "these people always", "they are all",
                "articulate for a", "well-spoken for a",
                "crime-ridden community", "dangerous neighborhood",
                "foreigners are", "immigrants always",
                "immigrants are responsible", "not from here",
            ],
            "stereotype_patterns": [
                r"(all|most|typical)\s+\w+\s+(people|individuals)\s+(are|tend)",
                r"(?:black|white|asian|hispanic|arab)\s+people\s+(?:are|always|never|tend|will)",
                r"immigrants?\s+(?:are|always|never|will|take)",
            ],
            "weight": 0.9
        },
        "age": {
            "keywords": [
                "senile", "decrepit", "over the hill", "out of touch",
                "entitled millennials", "lazy gen-z", "ok boomer",
                "young and naive", "too old to learn", "elderly confusion",
                "millennials are lazy", "gen z is", "boomers don't",
                "past their prime",
            ],
            "stereotype_patterns": [
                r"(old|young)\s+people\s+(always|never|can't)",
                r"(?:old|young|elderly)\s+people\s+(?:are|always|never|can'?t)",
            ],
            "weight": 0.7
        },
        "socioeconomic": {
            "keywords": [
                "trailer trash", "ghetto", "privileged", "elitist",
                "uneducated masses", "backwards", "uncivilized",
                "welfare queen", "trust fund", "poverty is a choice",
                "poor people are lazy", "if they just worked harder",
                "lower class mentality", "living off the government",
            ],
            "stereotype_patterns": [
                r"(?:poor|rich)\s+people\s+(?:are|always|deserve|should)",
            ],
            "weight": 0.75
        }
    }

    # ── Ideological bias indicators ──
    IDEOLOGICAL_BIAS = {
        "political_left": [
            "libtard", "snowflake", "radical left", "socialist agenda",
            "woke mob", "cancel culture extremists", "marxist plot",
            "left-wing propaganda", "far-left extremist",
        ],
        "political_right": [
            "fascist", "nazi", "alt-right", "racist conservatives",
            "bigoted right-wing", "extremist republicans",
            "right-wing extremist", "maga extremist",
        ],
        "absolutist_language": [
            "always", "never", "everyone knows", "obviously",
            "undeniably", "without exception", "all experts agree",
            "nobody can deny", "it's obvious that", "no one can argue",
            "completely proven", "universally accepted that",
        ]
    }

    # ── Toxicity patterns ──
    TOXICITY_PATTERNS = [
        (r"(stupid|idiotic|moronic)\s+\w+", 0.6),
        (r"(hate|despise|loathe)\s+(all|every)\s+\w+", 0.8),
        (r"(should be|deserve to be)\s+(eliminated|removed|punished)", 0.9),
        (r"(inferior|superior)\s+(race|gender|people)", 0.95),
        (r"disgusting\s+people|filthy\s+immigrants|scum\s+of\s+society", 0.95),
        (r"(subhuman|vermin|parasite)\b", 0.95),
    ]

    def __init__(self, use_perspective_api: bool = False):
        self.use_perspective_api = use_perspective_api
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled_demographic = {}
        for category, config in self.DEMOGRAPHIC_BIAS.items():
            self.compiled_demographic[category] = {
                "patterns": [re.compile(p, re.IGNORECASE) for p in config["stereotype_patterns"]],
                "keywords": config["keywords"],
                "weight": config["weight"]
            }
        self.compiled_toxicity = [
            (re.compile(p, re.IGNORECASE), w) for p, w in self.TOXICITY_PATTERNS
        ]

    def detect(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Same signature as original BiasDetector.detect().
        Returns same structure as original.
        """
        findings = []
        category_scores = defaultdict(float)
        total_bias_score = 0.0

        text_lower = text.lower()
        word_count = max(len(text.split()), 1)

        # 1. Demographic bias detection (WinoBias + StereoSet derived)
        for category, config in self.compiled_demographic.items():
            category_findings = []
            for keyword in config["keywords"]:
                if keyword.lower() in text_lower:
                    category_findings.append({
                        "type": "demographic_keyword",
                        "category": category,
                        "indicator": keyword,
                        "severity": config["weight"]
                    })
            for pattern in config["patterns"]:
                matches = pattern.findall(text)
                for match in matches:
                    match_text = " ".join(match) if isinstance(match, tuple) else match
                    category_findings.append({
                        "type": "stereotype_pattern",
                        "category": category,
                        "indicator": match_text[:50],
                        "severity": config["weight"]
                    })
            if category_findings:
                findings.extend(category_findings)
                category_scores[category] = min(len(category_findings) * config["weight"] / 3, 1.0)

        # 2. Ideological bias detection
        for ideology, keywords in self.IDEOLOGICAL_BIAS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    findings.append({
                        "type": "ideological_bias",
                        "category": ideology,
                        "indicator": keyword,
                        "severity": 0.7
                    })
                    category_scores["ideological"] += 0.15

        # 3. Toxicity pattern detection
        for pattern, weight in self.compiled_toxicity:
            matches = pattern.findall(text)
            for match in matches:
                findings.append({
                    "type": "toxicity",
                    "category": "toxic_language",
                    "indicator": match[:40] if isinstance(match, str) else str(match)[:40],
                    "severity": weight
                })
                total_bias_score += weight * 0.3

        # 4. Absolutist language check
        absolutist_count = sum(1 for term in self.IDEOLOGICAL_BIAS["absolutist_language"]
                               if term.lower() in text_lower)
        if absolutist_count > 2:
            findings.append({
                "type": "absolutist_language",
                "category": "rhetorical_bias",
                "indicator": f"{absolutist_count} absolutist terms detected",
                "severity": 0.4
            })
            total_bias_score += 0.15

        # Calculate final score (higher = less biased = better)
        if findings:
            avg_severity = sum(f["severity"] for f in findings) / len(findings)
            density_factor = min(len(findings) / (word_count / 100), 1.0)
            raw_bias = (avg_severity * 0.6 + density_factor * 0.4)
            score = max(0, round((1 - raw_bias) * 100))
        else:
            score = 95

        # Verdict
        if score >= 80:
            verdict = "Neutral"
        elif score >= 50:
            verdict = "Mild Bias Detected"
        else:
            verdict = "Biased Content"

        findings_summary = self._generate_findings_summary(findings, category_scores)

        return {
            "score": score,
            "verdict": verdict,
            "findings": findings_summary,
            "details": {
                "total_indicators": len(findings),
                "category_scores": dict(category_scores),
                "findings_list": findings[:10]
            }
        }

    def _generate_findings_summary(self, findings: List[Dict], category_scores: Dict) -> str:
        if not findings:
            return "No significant bias indicators detected. Content appears neutral and balanced."
        parts = []
        total = len(findings)
        categories_found = set(f.get("category") for f in findings)
        if "gender" in categories_found:
            parts.append("gender-related bias patterns")
        if "racial" in categories_found:
            parts.append("racial/ethnic bias indicators")
        if "ideological" in category_scores and category_scores["ideological"] > 0.2:
            parts.append("ideological language")
        if any(f["type"] == "toxicity" for f in findings):
            parts.append("toxic language patterns")
        if parts:
            return f"Detected {total} bias indicator(s): {', '.join(parts)}."
        return f"Detected {total} minor bias indicator(s) that may affect content neutrality."