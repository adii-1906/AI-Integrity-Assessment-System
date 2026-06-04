from flask import Flask
from flask_cors import CORS
from config import config


def create_app() -> Flask:
    """Application factory function."""
    app = Flask(__name__)

    CORS(app, resources={
        r"/api/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

    app.config.from_object(config)

    # Existing blueprints
    from routes.evaluate import evaluate_bp
    app.register_blueprint(evaluate_bp, url_prefix="/api")

    from routes.media import media_bp
    app.register_blueprint(media_bp, url_prefix="/api")

    # NEW — image deepfake detection
    from routes.image_deepfake import image_deepfake_bp
    app.register_blueprint(image_deepfake_bp, url_prefix="/api")

    @app.route("/health", methods=["GET"])
    def health_check():
        return {
            "status": "healthy",
            "service": "AICES - AI Content Integrity Evaluation System",
            "version": "3.0",
            "modules": ["M1-Bias", "M2-Hallucination", "M3-Privacy",
                        "M4-Explainability", "M5-Deepfake"],
            "endpoints": {
                "text":           "POST /api/evaluate",
                "media_extract":  "POST /api/extract-media",
                "media_evaluate": "POST /api/evaluate-media",
                "image_deepfake": "POST /api/analyze-image-deepfake",
                "modules":        "GET  /api/modules",
            }
        }

    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("  AICES v3.0 - AI Content Integrity Evaluation System")
    print("  Endpoints:")
    print("    POST /api/evaluate               — Text evaluation")
    print("    POST /api/extract-media          — OCR extraction")
    print("    POST /api/evaluate-media         — Extract + evaluate")
    print("    POST /api/analyze-image-deepfake — Image AI detection")
    print("    GET  /api/modules                — Module info")
    print("    GET  /health                     — Health check")
    print("=" * 60)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )