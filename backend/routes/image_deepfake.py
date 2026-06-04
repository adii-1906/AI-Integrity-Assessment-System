"""
AICES — Image Deepfake Detection Route
backend/routes/image_deepfake.py

New endpoint: POST /api/analyze-image-deepfake
Accepts: base64 image, returns full manipulation analysis
"""

from flask import Blueprint, request, jsonify
from http import HTTPStatus
import base64
import time

from services.image_deepfake_detector import ImageDeepfakeDetector

image_deepfake_bp = Blueprint("image_deepfake", __name__)
detector = ImageDeepfakeDetector()


@image_deepfake_bp.route("/analyze-image-deepfake", methods=["POST"])
def analyze_image_deepfake():
    """
    Analyze an uploaded image for AI generation / manipulation.

    Request JSON:
    {
        "file_data": "<base64 encoded image>",
        "filename":  "photo.jpg",
        "mime_type": "image/jpeg"
    }

    Response JSON:
    {
        "success": true,
        "verdict": "AI-Generated / Heavily Manipulated",
        "verdict_code": "ai_generated",
        "manipulation_percentage": 72.4,
        "image_info": { "width": 1024, "height": 768, ... },
        "technique_scores": {
            "ela":  { "name": "Error Level Analysis", "score": 68.2, ... },
            "dct":  { "name": "DCT Frequency Analysis", "score": 75.0, ... },
            "pna":  { "name": "Pixel Noise Analysis", "score": 55.1, ... }
        },
        "region_analysis": {
            "grid": "3x3",
            "total_regions": 9,
            "suspicious_regions": 5,
            "pct_image_suspicious": 55.6,
            "regions": [
                { "name": "Top-Left", "manipulation_pct": 82.3, "status": "suspicious" },
                ...
            ]
        },
        "heatmap_base64": "<base64 PNG heatmap>",
        "explanation": "This image shows strong signs of AI generation...",
        "processing_time_seconds": 1.23
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), HTTPStatus.BAD_REQUEST

        file_data = data.get("file_data", "")
        filename  = data.get("filename", "image")
        mime_type = data.get("mime_type", "image/jpeg")

        if not file_data:
            return jsonify({"error": "file_data (base64) is required"}), HTTPStatus.BAD_REQUEST

        # Validate it's an image
        supported = {"image/jpeg", "image/jpg", "image/png",
                     "image/webp", "image/gif", "image/bmp"}
        if mime_type.lower() not in supported:
            return jsonify({
                "error": f"Unsupported type: {mime_type}. Supported: JPEG, PNG, WebP, GIF, BMP"
            }), HTTPStatus.BAD_REQUEST

        # Decode base64
        try:
            image_bytes = base64.b64decode(file_data)
        except Exception:
            return jsonify({"error": "Invalid base64 data"}), HTTPStatus.BAD_REQUEST

        # Size check — max 10MB
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > 10:
            return jsonify({
                "error": f"Image too large ({size_mb:.1f}MB). Maximum 10MB."
            }), HTTPStatus.BAD_REQUEST

        # Run analysis
        start = time.time()
        result = detector.analyze(image_bytes, filename)
        elapsed = round(time.time() - start, 2)

        if not result.get("success"):
            return jsonify(result), HTTPStatus.UNPROCESSABLE_ENTITY

        result["processing_time_seconds"] = elapsed
        return jsonify(result), HTTPStatus.OK

    except Exception as e:
        return jsonify({
            "error": f"Image deepfake analysis failed: {str(e)}"
        }), HTTPStatus.INTERNAL_SERVER_ERROR