"""
AICES Debug Script — Run this to diagnose the 'list has no attribute items' error.
Run from the backend/ directory: python debug_error.py
"""

import os, json, sys

print("=" * 60)
print("AICES Error Diagnosis")
print("=" * 60)

# Check 1: knowledge_base.json format
kb_paths = [
    "data/knowledge_base.json",
    "../data/knowledge_base.json",
]
for path in kb_paths:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        dtype = type(data).__name__
        print(f"\n[CHECK 1] data/knowledge_base.json found")
        print(f"  Type: {dtype}")
        if isinstance(data, list):
            print("  ❌ BUG: This is a LIST, not a dict.")
            print("  FIX: Delete data/knowledge_base.json or convert it to dict format.")
        else:
            print("  ✓ Format is correct (dict)")
        break
else:
    print("\n[CHECK 1] data/knowledge_base.json — NOT FOUND (will use built-in KB)")
    print("  ✓ This is fine")

# Check 2: models directory
print("\n[CHECK 2] Checking models/ directory...")
if os.path.exists("models"):
    files = os.listdir("models")
    expected = [
        "m1_bias_model.pkl", "m1_bias_vectorizer.pkl",
        "m2_hallucination_model.pkl", "m2_hallucination_vectorizer.pkl",
        "m3_pii_config.json",
        "m4_explain_model.pkl", "m4_explain_vectorizer.pkl",
        "m5_deepfake_model.pkl", "m5_deepfake_tfidf.pkl",
    ]
    for f in expected:
        status = "✓" if f in files else "❌ MISSING — run train_models.py"
        print(f"  {status} {f}")
else:
    print("  ❌ models/ directory does not exist")
    print("  FIX: Run train_models.py first")

# Check 3: services import test
print("\n[CHECK 3] Testing service imports...")
sys.path.insert(0, ".")
services = [
    "services.bias_detection",
    "services.hallucination_detection", 
    "services.privacy_audit",
    "services.explainability",
    "services.deepfake_detection",
    "services.aggregator",
    "services.llm_handler",
]
for svc in services:
    try:
        __import__(svc)
        print(f"  ✓ {svc}")
    except Exception as e:
        print(f"  ❌ {svc}: {e}")

# Check 4: Quick evaluation test
print("\n[CHECK 4] Running quick evaluation test...")
try:
    from services.bias_detection import BiasDetector
    from services.hallucination_detection import HallucinationDetector
    from services.privacy_audit import PrivacyAuditor
    from services.explainability import ExplainabilityEngine
    from services.deepfake_detection import DeepfakeDetector
    from services.aggregator import IntegrityAggregator

    text = "The study found that approximately 65 percent of participants showed improvement based on published research."
    
    m1 = BiasDetector().detect(text)
    print(f"  ✓ M1 Bias: score={m1['score']}")
    
    m2 = HallucinationDetector().detect(text)
    print(f"  ✓ M2 Hallucination: score={m2['score']}")
    
    m3 = PrivacyAuditor().audit(text)
    print(f"  ✓ M3 Privacy: score={m3['score']}")
    
    m4 = ExplainabilityEngine().analyze(text)
    print(f"  ✓ M4 Explainability: score={m4['score']}")
    
    m5 = DeepfakeDetector().detect(text)
    print(f"  ✓ M5 Deepfake: score={m5['score']}")
    
    ii = IntegrityAggregator().compute_integrity_index({
        "m1_bias": m1["score"], "m2_hallucination": m2["score"],
        "m3_privacy": m3["score"], "m4_explainability": m4["score"],
        "m5_deepfake": m5["score"]
    })
    print(f"  ✓ Integrity Index: {ii['score']} ({ii['tier']})")
    print("\n✅ All services working correctly!")

except Exception as e:
    import traceback
    print(f"\n❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\nThis is the exact location of your bug.")

print("=" * 60)