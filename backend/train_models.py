"""
AICES — Complete Model Training Script
Run this ONCE to train all 5 ML models from your downloaded datasets.
Saves trained models to backend/models/ folder.

Usage:
    cd backend
    python train_models.py

Datasets needed in backend/datasets/:
    - winobias_pro.txt          (for M1 Bias)
    - stereoset (HuggingFace)   (for M1 Bias - downloaded via code)
    - fever_train.jsonl         (for M2 Hallucination)
    - fever_dev.jsonl           (for M2 Hallucination)
    - emails.csv                (for M3 Privacy validation)
    - esnli (HuggingFace)       (for M4 Explainability - downloaded via code)
    - hc3 (HuggingFace)         (for M5 Deepfake - downloaded via code)
"""

import os
import json
import pickle
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score

# ── Create models directory ──
os.makedirs("models", exist_ok=True)
print("=" * 60)
print("AICES MODEL TRAINING SCRIPT")
print("=" * 60)


# ════════════════════════════════════════════════
# MODULE 1 — BIAS DETECTION
# Dataset: WinoBias (txt) + StereoSet (HuggingFace)
# Model: Logistic Regression on TF-IDF features
# ════════════════════════════════════════════════

def train_m1_bias():
    print("\n[M1] Training Bias Detection Model...")
    print("      Loading WinoBias dataset...")

    texts = []
    labels = []

    # ── Load WinoBias from your datasets folder ──
    winobias_path = "datasets/winobias_pro.txt"
    if os.path.exists(winobias_path):
        with open(winobias_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(1)  # All WinoBias = stereotype (biased)
        print(f"      WinoBias loaded: {len(texts)} samples")
    else:
        print("      WARNING: winobias_pro.txt not found, using synthetic data")

    # ── Load StereoSet from HuggingFace ──
    print("      Loading StereoSet from HuggingFace...")
    try:
        from datasets import load_dataset
        stereoset = load_dataset("stereoset", "intrasentence", trust_remote_code=True)

        for item in stereoset["validation"]:
            sentences = item["sentences"]["sentence"]
            gold_labels = item["sentences"]["gold_label"]
            for sent, gold in zip(sentences, gold_labels):
                texts.append(sent)
                # gold_label: 0=anti-stereotype, 1=stereotype, 2=unrelated
                labels.append(1 if gold == 1 else 0)

        print(f"      StereoSet loaded: total {len(texts)} samples")
    except Exception as e:
        print(f"      StereoSet failed: {e}")
        print("      Adding fallback bias examples...")
        # Fallback bias examples if HuggingFace unavailable
        bias_examples = [
            ("Women are too emotional for leadership roles.", 1),
            ("Immigrants are responsible for rising crime rates.", 1),
            ("The doctor finished his rounds carefully.", 0),
            ("She is an excellent engineer with great skills.", 0),
            ("Men are naturally better at math than women.", 1),
            ("The nurse helped the patient with her treatment.", 0),
            ("Poor people are lazy and don't want to work.", 1),
            ("The CEO made a strategic decision for the company.", 0),
            ("Black neighborhoods always have more crime.", 1),
            ("The researcher published her findings last week.", 0),
        ]
        for text, label in bias_examples * 100:
            texts.append(text)
            labels.append(label)

    if len(texts) == 0:
        print("      ERROR: No training data for M1")
        return False

    # ── Train TF-IDF + Logistic Regression ──
    print(f"      Training on {len(texts)} samples...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"      M1 F1-Score: {f1:.3f}")

    # ── Save model ──
    pickle.dump(model, open("models/m1_bias_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/m1_bias_vectorizer.pkl", "wb"))
    print("      M1 model saved to models/m1_bias_model.pkl")
    return True


# ════════════════════════════════════════════════
# MODULE 2 — HALLUCINATION DETECTION
# Dataset: FEVER (fever_train.jsonl + fever_dev.jsonl)
# Model: Random Forest on TF-IDF features
# ════════════════════════════════════════════════

def train_m2_hallucination():
    print("\n[M2] Training Hallucination Detection Model...")
    print("      Loading FEVER dataset...")

    claims = []
    labels = []

    # ── Load FEVER train ──
    fever_train_path = "datasets/fever_train.jsonl"
    if os.path.exists(fever_train_path):
        with open(fever_train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 60000:  # Use 60K samples for speed
                    break
                try:
                    item = json.loads(line.strip())
                    if "claim" in item and "label" in item:
                        claims.append(item["claim"])
                        # SUPPORTS = not hallucinated (0)
                        # REFUTES or NOT ENOUGH INFO = hallucinated (1)
                        label = 0 if item["label"] == "SUPPORTS" else 1
                        labels.append(label)
                except:
                    continue
        print(f"      FEVER train loaded: {len(claims)} samples")
    else:
        print("      WARNING: fever_train.jsonl not found")

    # ── Load FEVER dev ──
    fever_dev_path = "datasets/fever_dev.jsonl"
    if os.path.exists(fever_dev_path):
        with open(fever_dev_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "claim" in item and "label" in item:
                        claims.append(item["claim"])
                        label = 0 if item["label"] == "SUPPORTS" else 1
                        labels.append(label)
                except:
                    continue
        print(f"      FEVER total loaded: {len(claims)} samples")

    if len(claims) == 0:
        print("      WARNING: No FEVER data found, using fallback examples")
        fallback = [
            ("The Earth is flat and has no curvature.", 1),
            ("Barack Obama was the 44th President of the United States.", 0),
            ("Water boils at 150 degrees Celsius at sea level.", 1),
            ("The Eiffel Tower is located in Paris, France.", 0),
            ("According to Harvard scientists, coffee cures all diseases.", 1),
            ("The moon orbits around the Earth.", 0),
            ("Studies show 100% of people benefit from this treatment.", 1),
            ("World War II ended in 1945.", 0),
            ("Napoleon Bonaparte was born in France in 1769.", 0),
            ("This medication is completely safe with no side effects.", 1),
        ]
        for text, label in fallback * 200:
            claims.append(text)
            labels.append(label)

    # ── Train TF-IDF + Random Forest ──
    print(f"      Training Random Forest on {len(claims)} claims...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(claims)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      M2 Accuracy: {acc:.3f}")

    # ── Save ──
    pickle.dump(model, open("models/m2_hallucination_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/m2_hallucination_vectorizer.pkl", "wb"))
    print("      M2 model saved to models/m2_hallucination_model.pkl")
    return True


# ════════════════════════════════════════════════
# MODULE 3 — PRIVACY AUDIT
# Dataset: emails.csv (Enron) + spaCy NER
# Model: Regex patterns (12 classes) + spaCy NER
# ════════════════════════════════════════════════

def train_m3_privacy():
    print("\n[M3] Setting up Privacy Audit (Regex + spaCy NER)...")

    # ── Validate regex patterns against Enron emails ──
    emails_path = "datasets/emails.csv"
    if os.path.exists(emails_path):
        print("      Loading Enron email dataset for validation...")
        try:
            # Enron CSV has 'message' column
            df = pd.read_csv(emails_path, nrows=2000,
                             usecols=["message"], on_bad_lines="skip")
            df = df.dropna()
            print(f"      Loaded {len(df)} emails for PII pattern validation")

            # Test all 12 PII patterns
            pii_patterns = {
                "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "phone_us":    r"\b\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b",
                "ssn":         r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                "ip_address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                "date_of_birth": r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
            }

            found_counts = {k: 0 for k in pii_patterns}
            for text in df["message"].astype(str):
                for pii_type, pattern in pii_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        found_counts[pii_type] += 1

            print("      PII detection rates on Enron emails:")
            for pii_type, count in found_counts.items():
                rate = count / len(df) * 100
                print(f"        {pii_type}: found in {rate:.1f}% of emails")

        except Exception as e:
            print(f"      Email validation skipped: {e}")
    else:
        print("      WARNING: emails.csv not found — skipping Enron validation")

    # ── Test spaCy NER ──
    print("      Testing spaCy NER...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        test = "John Smith from London called +44 7911 123456 about order #12345."
        doc = nlp(test)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"      spaCy NER working: found {entities}")
        # Save a marker file indicating spaCy is available
        with open("models/m3_spacy_available.txt", "w") as f:
            f.write("yes")
    except Exception as e:
        print(f"      spaCy not available: {e}")
        print("      Run: python -m spacy download en_core_web_sm")
        with open("models/m3_spacy_available.txt", "w") as f:
            f.write("no")

    # Save PII pattern configuration (the "trained" component for M3)
    pii_config = {
        "patterns": {
            "ssn":          {"regex": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                             "risk_level": "critical", "weight": 0.95},
            "aadhaar":      {"regex": r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
                             "risk_level": "critical", "weight": 0.95},
            "credit_card":  {"regex": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                             "risk_level": "critical", "weight": 0.98},
            "bank_account": {"regex": r"\b\d{9,18}\b(?=.*\baccount\b)",
                             "risk_level": "critical", "weight": 0.95},
            "passport":     {"regex": r"\b[A-Z]{1,2}\d{6,9}\b",
                             "risk_level": "high", "weight": 0.80},
            "medical_rec":  {"regex": r"\b(?:MRN|MR|patient.?id|record.?no)[\s:]*\d+\b",
                             "risk_level": "high", "weight": 0.85},
            "email":        {"regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                             "risk_level": "medium", "weight": 0.50},
            "phone_us":     {"regex": r"\b\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b",
                             "risk_level": "medium", "weight": 0.50},
            "phone_india":  {"regex": r"(?:\+91[\s\-]?)?[6-9]\d{9}\b",
                             "risk_level": "medium", "weight": 0.50},
            "dob":          {"regex": r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
                             "risk_level": "medium", "weight": 0.60},
            "ip_address":   {"regex": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
                             "risk_level": "low", "weight": 0.30},
            "sensitive_kw": {"regex": r"\b(?:password|passwd|api.?key|secret|token|cvv|pin|private.?key|auth.?token)\b",
                             "risk_level": "medium", "weight": 0.40},
        },
        "risk_weights": {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
    }
    with open("models/m3_pii_config.json", "w") as f:
        json.dump(pii_config, f, indent=2)
    print("      M3 PII config saved to models/m3_pii_config.json")
    return True


# ════════════════════════════════════════════════
# MODULE 4 — EXPLAINABILITY ANALYSIS
# Dataset: e-SNLI (HuggingFace)
# Model: LinearSVC on TF-IDF features
# ════════════════════════════════════════════════

def train_m4_explainability():
    print("\n[M4] Training Explainability Model...")
    print("      Loading e-SNLI dataset from HuggingFace...")

    texts = []
    labels = []

    try:
        from datasets import load_dataset
        esnli = load_dataset("esnli", trust_remote_code=True)

        def score_explanation(exp):
            """Label explanation quality: 1=high quality, 0=low quality"""
            if not exp or len(exp) < 10:
                return 0
            exp_lower = exp.lower()
            positive = ["because", "therefore", "since", "as a result",
                        "due to", "hence", "thus", "for example",
                        "such as", "according to", "based on",
                        "suggests", "indicates", "implies",
                        "this means", "in other words"]
            negative = ["obviously", "clearly", "everyone knows",
                        "trust me", "always", "never", "definitely",
                        "certainly", "undeniably", "it is obvious"]
            pos_count = sum(1 for w in positive if w in exp_lower)
            neg_count = sum(1 for w in negative if w in exp_lower)
            return 1 if (pos_count >= 1 and neg_count == 0) else 0

        count = 0
        for item in esnli["train"]:
            if count >= 25000:
                break
            exp = item.get("explanation_1", "")
            if exp and len(exp) > 15:
                combined = (item.get("premise", "") + " " +
                            item.get("hypothesis", "") + " " + exp)
                texts.append(combined)
                labels.append(score_explanation(exp))
                count += 1

        print(f"      e-SNLI loaded: {len(texts)} samples")
        print(f"      High-quality: {sum(labels)}, Low-quality: {len(labels)-sum(labels)}")

    except Exception as e:
        print(f"      e-SNLI failed: {e}")
        print("      Using fallback explainability examples...")
        fallback = [
            ("The answer is correct because water freezes at 0 degrees Celsius.", 1),
            ("Obviously this is true everyone knows it.", 0),
            ("Therefore based on the evidence the conclusion follows.", 1),
            ("Trust me this will definitely work perfectly.", 0),
            ("According to the study results suggest a moderate effect.", 1),
            ("This is clearly wrong always and never any exceptions.", 0),
            ("For example the data shows a 15 percent increase.", 1),
            ("It is common knowledge so no explanation needed.", 0),
            ("Since the temperature rose the ice began to melt.", 1),
            ("Everyone knows this is undeniably true.", 0),
        ]
        for text, label in fallback * 300:
            texts.append(text)
            labels.append(label)

    if len(texts) == 0:
        print("      ERROR: No training data for M4")
        return False

    # ── Train LinearSVC (fast and effective for text) ──
    print(f"      Training LinearSVC on {len(texts)} samples...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CalibratedClassifierCV wraps SVC to give probability scores
    base_svc = LinearSVC(max_iter=2000, C=1.0, class_weight="balanced")
    model = CalibratedClassifierCV(base_svc)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      M4 Accuracy: {acc:.3f}")

    # ── Save ──
    pickle.dump(model, open("models/m4_explain_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/m4_explain_vectorizer.pkl", "wb"))
    print("      M4 model saved to models/m4_explain_model.pkl")
    return True


# ════════════════════════════════════════════════
# MODULE 5 — DEEPFAKE DETECTION
# Dataset: HC3 (HuggingFace)
# Model: Gradient Boosting on TF-IDF + Statistical Features
# ════════════════════════════════════════════════

def extract_statistical_features(text):
    """Extract 7 statistical ML features for deepfake detection"""
    if not text or len(text) < 10:
        return [0.5, 5.0, 0.0, 50, 30, 0.0, 0.3]

    words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # 1. Type-Token Ratio (vocabulary diversity)
    ttr = len(set(words)) / len(words) if words else 0.5

    # 2. Sentence length variance
    sent_lengths = [len(s.split()) for s in sentences]
    variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 5.0

    # 3. Bigram repetition ratio
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_repeat = ((len(bigrams) - len(set(bigrams))) / len(bigrams)
                     if bigrams else 0.0)

    # 4. Total word count
    word_count = len(words)

    # 5. Unique word count
    unique_count = len(set(words))

    # 6. Long word ratio (words > 7 chars = formal/AI writing)
    long_words = sum(1 for w in words if len(w) > 7)
    long_ratio = long_words / len(words) if words else 0.0

    # 7. Average sentence length
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 15.0

    return [ttr, variance, bigram_repeat, word_count,
            unique_count, long_ratio, avg_sent_len]


def train_m5_deepfake():
    print("\n[M5] Training Deepfake Detection Model...")
    print("      Loading HC3 dataset from HuggingFace...")

    texts = []
    labels = []
    stat_features = []

    try:
        from datasets import load_dataset
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)

        count = 0
        for item in hc3["train"]:
            if count >= 30000:
                break
            # Human answers = label 0
            for ans in item.get("human_answers", []):
                if ans and len(ans) > 50:
                    texts.append(ans[:500])
                    labels.append(0)
                    stat_features.append(extract_statistical_features(ans))
                    count += 1
            # ChatGPT answers = label 1
            for ans in item.get("chatgpt_answers", []):
                if ans and len(ans) > 50:
                    texts.append(ans[:500])
                    labels.append(1)
                    stat_features.append(extract_statistical_features(ans))
                    count += 1

        print(f"      HC3 loaded: {len(texts)} samples")
        print(f"      Human: {labels.count(0)}, AI: {labels.count(1)}")

    except Exception as e:
        print(f"      HC3 failed: {e}")
        print("      Using fallback deepfake examples...")
        human_examples = [
            "I think this is really interesting and I loved how it all came together honestly.",
            "Yeah I'm not sure about that, it seems kinda off to me personally.",
            "Wow this is amazing! Can't believe how good it turned out.",
            "Honestly I've been struggling with this for weeks now it's frustrating.",
            "I dunno, maybe? It's hard to say without more context lol.",
        ]
        ai_examples = [
            "It is important to note that this process involves several key considerations.",
            "In conclusion, leveraging these strategies can facilitate optimal outcomes.",
            "Furthermore, it is worth mentioning that implementation requires careful consideration.",
            "This comprehensive approach enables organizations to effectively optimize performance.",
            "It is important to understand that this methodology provides significant benefits.",
        ]
        for text in human_examples * 400:
            texts.append(text)
            labels.append(0)
            stat_features.append(extract_statistical_features(text))
        for text in ai_examples * 400:
            texts.append(text)
            labels.append(1)
            stat_features.append(extract_statistical_features(text))

    if len(texts) == 0:
        print("      ERROR: No training data for M5")
        return False

    # ── Build combined feature matrix ──
    print(f"      Building features for {len(texts)} samples...")
    tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vec.fit_transform(texts)
    X_stats = np.array(stat_features)

    # Combine TF-IDF + statistical features
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_stats)])

    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train Gradient Boosting ──
    print("      Training Gradient Boosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train.toarray(), y_train)

    y_proba = model.predict_proba(X_test.toarray())[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"      M5 ROC-AUC: {auc:.3f}")

    # ── Save ──
    pickle.dump(model, open("models/m5_deepfake_model.pkl", "wb"))
    pickle.dump(tfidf_vec, open("models/m5_deepfake_tfidf.pkl", "wb"))
    print("      M5 model saved to models/m5_deepfake_model.pkl")
    return True


# ════════════════════════════════════════════════
# MAIN — Run all training
# ════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    results["M1_Bias"]            = train_m1_bias()
    results["M2_Hallucination"]   = train_m2_hallucination()
    results["M3_Privacy"]         = train_m3_privacy()
    results["M4_Explainability"]  = train_m4_explainability()
    results["M5_Deepfake"]        = train_m5_deepfake()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    for module, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {module}: {status}")

    print("\nModels saved in backend/models/:")
    for f in os.listdir("models"):
        size = os.path.getsize(f"models/{f}")
        print(f"  {f}  ({size/1024:.1f} KB)")

    print("\nNext step: Run the Flask app with trained models:")
    print("  python app.py")