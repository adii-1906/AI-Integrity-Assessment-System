"""
AICES Media Upload Route
Handles image/video upload, extracts text, then evaluates via existing pipeline.
Add this to routes/ as media.py and register in app.py.
"""

from flask import Blueprint, request, jsonify
from http import HTTPStatus
import time

from services.media_extract import MediaExtractor
from services.bias_detection import BiasDetector
from services.hallucination_detection import HallucinationDetector
from services.privacy_audit import PrivacyAuditor
from services.explainability import ExplainabilityEngine
from services.deepfake_detection import DeepfakeDetector
from services.aggregator import IntegrityAggregator
from services.llm_handler import LLMHandler

media_bp = Blueprint("media", __name__)

# Reuse existing services
extractor = MediaExtractor()
bias_detector = BiasDetector()
hallucination_detector = HallucinationDetector()
privacy_auditor = PrivacyAuditor()
explainability_engine = ExplainabilityEngine()
deepfake_detector = DeepfakeDetector()
aggregator = IntegrityAggregator()
llm_handler = LLMHandler()


@media_bp.route("/extract-media", methods=["POST"])
def extract_media():
    """
    Step 1: Extract text from uploaded image/video.
    Frontend can preview extracted text before running full evaluation.

    Request JSON:
        { "file_data": "<base64>", "mime_type": "image/jpeg", "filename": "photo.jpg" }

    Response:
        { "success": true, "text": "...", "method": "...", "media_type": "image", "details": {...} }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), HTTPStatus.BAD_REQUEST

        file_data = data.get("file_data", "")
        mime_type = data.get("mime_type", "")
        filename = data.get("filename", "")

        if not file_data:
            return jsonify({"error": "file_data (base64) is required"}), HTTPStatus.BAD_REQUEST
        if not mime_type:
            return jsonify({"error": "mime_type is required"}), HTTPStatus.BAD_REQUEST

        result = extractor.extract(file_data, mime_type, filename)
        return jsonify(result), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": f"Media extraction failed: {str(e)}"}), HTTPStatus.INTERNAL_SERVER_ERROR


@media_bp.route("/evaluate-media", methods=["POST"])
def evaluate_media():
    """
    Step 2 (combined): Extract text from media AND run full AICES evaluation.

    Request JSON:
        { "file_data": "<base64>", "mime_type": "image/jpeg", "filename": "photo.jpg", "content_type": "text" }

    Response: Same structure as /evaluate + media_extraction field
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), HTTPStatus.BAD_REQUEST

        file_data = data.get("file_data", "")
        mime_type = data.get("mime_type", "")
        filename = data.get("filename", "")
        content_type = data.get("content_type", "text")

        if not file_data or not mime_type:
            return jsonify({"error": "file_data and mime_type are required"}), HTTPStatus.BAD_REQUEST

        start_time = time.time()

        # Step 1: Extract text
        extraction = extractor.extract(file_data, mime_type, filename)
        if not extraction["success"]:
            return jsonify({
                "success": False,
                "error": extraction.get("error", "Text extraction failed"),
                "media_extraction": extraction
            }), HTTPStatus.UNPROCESSABLE_ENTITY

        content = extraction["text"]

        if len(content) < 30:
            return jsonify({
                "success": False,
                "error": f"Extracted text too short ({len(content)} chars). Need at least 30 characters for evaluation.",
                "media_extraction": extraction,
                "extracted_text": content
            }), HTTPStatus.UNPROCESSABLE_ENTITY

        # Step 2: Run all five modules (same as /evaluate)
        m1_result = bias_detector.detect(content, content_type)
        m2_result = hallucination_detector.detect(content)
        m3_result = privacy_auditor.audit(content)
        m4_result = explainability_engine.analyze(content)
        m5_result = deepfake_detector.detect(content, content_type)

        module_scores = {
            "m1_bias": m1_result["score"],
            "m2_hallucination": m2_result["score"],
            "m3_privacy": m3_result["score"],
            "m4_explainability": m4_result["score"],
            "m5_deepfake": m5_result["score"]
        }
        integrity_index = aggregator.compute_integrity_index(module_scores)

        # Step 3: Generate explanations
        def get_detail(module_name, result):
            score = result.get("score", 50)
            verdict = result.get("verdict", "")
            findings = result.get("findings", "")
            prompt = (f"Module: {module_name}\nScore: {score}/100\nVerdict: {verdict}\n"
                      f"Automated Findings: {findings}\nContent Type: {content_type}\n"
                      f"Content Preview: {content[:400]}\n\n"
                      "Write a detailed expert explanation of WHY this score was given, "
                      "WHAT issues were found, and WHAT the implications are. Prose only, no bullets.")
            try:
                resp = llm_handler.generate(prompt, max_tokens=300, temperature=0.4)
                return resp.get("response", "").strip() or findings
            except Exception:
                return findings

        m1_detail = get_detail("M1 Bias Detection", m1_result)
        m2_detail = get_detail("M2 Hallucination Detection", m2_result)
        m3_detail = get_detail("M3 Privacy Audit", m3_result)
        m4_detail = get_detail("M4 Explainability Analysis", m4_result)
        m5_detail = get_detail("M5 Deepfake/Synthetic Detection", m5_result)

        # Step 4: Narrative + flags
        ii_score = integrity_index["score"]
        if ii_score >= 85:
            narrative = f"**High-integrity content** extracted from media — Integrity Index **{ii_score}**."
        elif ii_score >= 65:
            narrative = f"**Moderate-integrity content** extracted from media — Integrity Index **{ii_score}** — some concerns identified."
        elif ii_score >= 40:
            narrative = f"**Low-integrity content** extracted from media — Integrity Index **{ii_score}** — significant issues."
        else:
            narrative = f"**Critical integrity failure** in media content — Integrity Index **{ii_score}**."

        if m2_result["score"] < 60:
            narrative += " **Hallucination risk** elevated in extracted text."
        if m1_result["score"] < 70:
            narrative += " **Bias indicators** detected."
        if m3_result["score"] < 80:
            narrative += " **Privacy concerns** identified."

        flags = []
        if m1_result["score"] < 50: flags.append("HIGH BIAS RISK")
        elif m1_result["score"] < 70: flags.append("MILD BIAS")
        if m2_result["score"] < 50: flags.append("HALLUCINATION DETECTED")
        elif m2_result["score"] < 70: flags.append("UNVERIFIED CLAIMS")
        if m3_result["score"] < 50: flags.append("HIGH PII RISK")
        elif m3_result["score"] < 80: flags.append("PII DETECTED")
        if m4_result["score"] < 60: flags.append("LOW EXPLAINABILITY")
        if m5_result["score"] < 50: flags.append("SYNTHETIC CONTENT")
        elif m5_result["score"] < 70: flags.append("AI-GENERATED MARKERS")

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "content_type": content_type,
            "integrity_index": integrity_index["score"],
            "trust_tier": integrity_index["tier"],
            "m1_bias": {"score": m1_result["score"], "verdict": m1_result["verdict"], "findings": m1_result["findings"], "detail": m1_detail},
            "m2_hallucination": {"score": m2_result["score"], "verdict": m2_result["verdict"], "findings": m2_result["findings"], "detail": m2_detail},
            "m3_privacy": {"score": m3_result["score"], "verdict": m3_result["verdict"], "findings": m3_result["findings"], "detail": m3_detail},
            "m4_explainability": {"score": m4_result["score"], "verdict": m4_result["verdict"], "findings": m4_result["findings"], "detail": m4_detail},
            "m5_deepfake": {"score": m5_result["score"], "verdict": m5_result["verdict"], "findings": m5_result["findings"], "detail": m5_detail},
            "narrative": narrative,
            "flags": flags,
            "processing_time_seconds": elapsed,
            "media_extraction": {
                "method": extraction.get("method"),
                "media_type": extraction.get("media_type"),
                "details": extraction.get("details", {}),
                "extracted_text_preview": content[:300] + ("..." if len(content) > 300 else ""),
                "extracted_text_full": content
            }
        }), HTTPStatus.OK

    except Exception as e:
        return jsonify({"error": f"Media evaluation failed: {str(e)}"}), HTTPStatus.INTERNAL_SERVER_ERROR