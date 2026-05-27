"""
AICES Evaluation Endpoint
Main API route for content integrity analysis.
Includes per-module detailed AI explanations via LLMHandler.
"""

from flask import Blueprint, request, jsonify
from http import HTTPStatus
import time

from services.bias_detection import BiasDetector
from services.hallucination_detection import HallucinationDetector
from services.privacy_audit import PrivacyAuditor
from services.explainability import ExplainabilityEngine
from services.deepfake_detection import DeepfakeDetector
from services.aggregator import IntegrityAggregator
from services.llm_handler import LLMHandler


evaluate_bp = Blueprint("evaluate", __name__)

# Initialize services
bias_detector = BiasDetector()
hallucination_detector = HallucinationDetector()
privacy_auditor = PrivacyAuditor()
explainability_engine = ExplainabilityEngine()
deepfake_detector = DeepfakeDetector()
aggregator = IntegrityAggregator()
llm_handler = LLMHandler()


@evaluate_bp.route("/evaluate", methods=["POST"])
def evaluate_content():
    """
    Main AICES evaluation endpoint.
    Returns integrity analysis across all five modules with detailed AI explanations.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), HTTPStatus.BAD_REQUEST

        content = data.get("content", "").strip()
        content_type = data.get("content_type", "text")

        if not content:
            return jsonify({"error": "Content field is required"}), HTTPStatus.BAD_REQUEST

        if len(content) < 30:
            return jsonify({"error": "Content too short. Please provide at least a few sentences."}), HTTPStatus.BAD_REQUEST

        start_time = time.time()

        # Run all five modules
        m1_result = bias_detector.detect(content, content_type)
        m2_result = hallucination_detector.detect(content)
        m3_result = privacy_auditor.audit(content)
        m4_result = explainability_engine.analyze(content)
        m5_result = deepfake_detector.detect(content, content_type)

        # Aggregate scores
        module_scores = {
            "m1_bias": m1_result["score"],
            "m2_hallucination": m2_result["score"],
            "m3_privacy": m3_result["score"],
            "m4_explainability": m4_result["score"],
            "m5_deepfake": m5_result["score"]
        }
        integrity_index = aggregator.compute_integrity_index(module_scores)

        # Generate per-module detailed AI explanations
        m1_detail = _generate_module_detail("M1 Bias Detection", m1_result, content, content_type)
        m2_detail = _generate_module_detail("M2 Hallucination Detection", m2_result, content, content_type)
        m3_detail = _generate_module_detail("M3 Privacy Audit", m3_result, content, content_type)
        m4_detail = _generate_module_detail("M4 Explainability Analysis", m4_result, content, content_type)
        m5_detail = _generate_module_detail("M5 Deepfake/Synthetic Detection", m5_result, content, content_type)

        # Generate overall narrative
        narrative = _generate_narrative(content_type, m1_result, m2_result, m3_result, m4_result, m5_result, integrity_index)

        # Collect flags
        flags = _collect_flags(m1_result, m2_result, m3_result, m4_result, m5_result)

        elapsed_time = round(time.time() - start_time, 2)

        response = {
            "success": True,
            "content_type": content_type,
            "integrity_index": integrity_index["score"],
            "trust_tier": integrity_index["tier"],
            "m1_bias": {
                "score": m1_result["score"],
                "verdict": m1_result["verdict"],
                "findings": m1_result["findings"],
                "detail": m1_detail
            },
            "m2_hallucination": {
                "score": m2_result["score"],
                "verdict": m2_result["verdict"],
                "findings": m2_result["findings"],
                "detail": m2_detail
            },
            "m3_privacy": {
                "score": m3_result["score"],
                "verdict": m3_result["verdict"],
                "findings": m3_result["findings"],
                "detail": m3_detail
            },
            "m4_explainability": {
                "score": m4_result["score"],
                "verdict": m4_result["verdict"],
                "findings": m4_result["findings"],
                "detail": m4_detail
            },
            "m5_deepfake": {
                "score": m5_result["score"],
                "verdict": m5_result["verdict"],
                "findings": m5_result["findings"],
                "detail": m5_detail
            },
            "narrative": narrative,
            "flags": flags,
            "processing_time_seconds": elapsed_time
        }

        return jsonify(response), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), HTTPStatus.INTERNAL_SERVER_ERROR


def _generate_module_detail(module_name: str, result: dict, content: str, content_type: str) -> str:
    """
    Generate a detailed AI explanation for a specific module's findings.
    """
    score = result.get("score", 50)
    verdict = result.get("verdict", "Unknown")
    findings = result.get("findings", "No findings available.")

    content_preview = content[:600] + "..." if len(content) > 600 else content

    prompt = f"""You are an expert AI content integrity analyst. Analyze the following module result and provide a detailed, insightful explanation in 3-5 sentences.

Module: {module_name}
Score: {score}/100
Verdict: {verdict}
Automated Findings: {findings}
Content Type: {content_type}
Content Preview: {content_preview}

Write a detailed expert explanation of WHY this score was given, WHAT specific issues were found in the text, and WHAT the implications are. Be specific about the actual content. Be direct and analytical. Do not use bullet points — write flowing prose only."""

    try:
        response = llm_handler.generate(prompt, max_tokens=300, temperature=0.4)
        detail = response.get("response", "").strip()
        if detail:
            return detail
    except Exception:
        pass

    # Fallback if LLM fails
    return findings


def _generate_narrative(content_type, m1, m2, m3, m4, m5, ii):
    """Generate overall narrative summary."""
    parts = []
    if ii["score"] >= 85:
        parts.append(f"**High-integrity content** detected with an Integrity Index of **{ii['score']}**.")
    elif ii["score"] >= 65:
        parts.append(f"**Moderate-integrity content** with an Integrity Index of **{ii['score']}** — some concerns identified.")
    elif ii["score"] >= 40:
        parts.append(f"**Low-integrity content** with an Integrity Index of **{ii['score']}** — significant issues detected.")
    else:
        parts.append(f"**Critical integrity failure** — Integrity Index of **{ii['score']}** indicates high risk.")

    if m2["score"] < 60:
        parts.append("**Hallucination risk** is elevated — several claims could not be verified against factual sources.")
    if m1["score"] < 70:
        parts.append("**Bias indicators** detected in the content — review for demographic or ideological imbalance.")
    if m3["score"] < 80:
        parts.append("**Privacy concerns** — potentially sensitive information detected.")
    if m5["score"] < 70:
        parts.append("**Synthetic content markers** identified — text patterns suggest AI generation.")

    return " ".join(parts)


def _collect_flags(m1, m2, m3, m4, m5):
    """Collect warning flags from all modules."""
    flags = []
    if m1["score"] < 50:
        flags.append("HIGH BIAS RISK")
    elif m1["score"] < 70:
        flags.append("MILD BIAS")
    if m2["score"] < 50:
        flags.append("HALLUCINATION DETECTED")
    elif m2["score"] < 70:
        flags.append("UNVERIFIED CLAIMS")
    if m3["score"] < 50:
        flags.append("HIGH PII RISK")
    elif m3["score"] < 80:
        flags.append("PII DETECTED")
    if m4["score"] < 60:
        flags.append("LOW EXPLAINABILITY")
    if m5["score"] < 50:
        flags.append("SYNTHETIC CONTENT")
    elif m5["score"] < 70:
        flags.append("AI-GENERATED MARKERS")
    return flags


@evaluate_bp.route("/modules", methods=["GET"])
def get_modules():
    """Return information about all AICES modules."""
    return jsonify({
        "modules": [
            {"id": "m1_bias", "name": "Bias Detection", "description": "Analyzes demographic and ideological bias", "weight": 0.20},
            {"id": "m2_hallucination", "name": "Hallucination Detection", "description": "Verifies factual claims against knowledge bases", "weight": 0.25},
            {"id": "m3_privacy", "name": "Privacy Audit", "description": "Detects PII including emails, phone numbers, SSN", "weight": 0.20},
            {"id": "m4_explainability", "name": "Explainability Analysis", "description": "Evaluates reasoning transparency", "weight": 0.15},
            {"id": "m5_deepfake", "name": "Deepfake Detection", "description": "Identifies synthetic content markers", "weight": 0.20}
        ]
    })