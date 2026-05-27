from flask import Flask
from flask_cors import CORS

from config import config
from routes.evaluate import evaluate_bp


def create_app() -> Flask:
    """Application factory function."""
    app = Flask(__name__)
    
    # Enable CORS for frontend integration
    CORS(app, resources={
        r"/api/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Load configuration
    app.config.from_object(config)
    
    # Register blueprints
    app.register_blueprint(evaluate_bp, url_prefix="/api")
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        return {
            "status": "healthy",
            "service": "AICES - AI Content Integrity Evaluation System",
            "version": "1.0",
            "modules": ["M1-Bias", "M2-Hallucination", "M3-Privacy", "M4-Explainability", "M5-Deepfake"]
        }
    
    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("  AICES - AI Content Integrity Evaluation System")
    print("  Starting server...")
    print(f"  URL: http://localhost:{config.PORT}")
    print("=" * 60)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
