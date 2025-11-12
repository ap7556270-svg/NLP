# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt
import io
import logging
from typing import Optional, Tuple, List
import os
from ftfy import fix_text

API_KEY = st.secrets["GOOGLE_FACTCHECK_API_KEY"]

# ----------------------------------------------
# Small cleaner using ftfy + whitespace collapse
# ----------------------------------------------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()

# ----------------------------------------
# Create a helper to normalize text labels
# ----------------------------------------

def normalize_label(label_text: str) -> str:
    """Convert label text into simplified True/False/Mixed."""
    if not label_text:
        return "Unknown"
    text = label_text.lower()
    if any(x in text for x in ["true", "accurate", "fact", "real"]):
        return "True"
    if any(x in text for x in ["false", "fake", "pants", "incorrect"]):
        return "False"
    if any(x in text for x in ["half", "mixed", "partly"]):
        return "Mixed"
    return "Unknown"

# ---------------------------
# Google Fact Check
# ---------------------------
def get_fact_check_results(query):
    """Fetch fact-check results for the given query from Google Fact Check Tools API."""
    if not API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        claims = data.get("claims", [])
        results = []
        for claim in claims:
            reviews = claim.get("claimReview", [])
            for r in reviews:
                results.append({
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "title": r.get("title", ""),
                    "rating": r.get("textualRating", "No Rating"),
                    "url": r.get("url", "")
                })
        return results
    except Exception as e:
        return [{"publisher": "Error", "title": str(e), "rating": "", "url": ""}]

# Imbalanced learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# NLP & ML
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------
# CONFIG
# ---------------------------
SCRAPED_DATA_PATH = "politifact_data.csv"
N_SPLITS = 5
MAX_PAGES = 100  # safety
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2  # seconds base

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# SpaCy loader (cached)
# ---------------------------
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error("SpaCy model 'en_core_web_sm' not found. Add the wheel URL to requirements.txt in your deploy environment.")
        st.code("""
# Example to add in requirements.txt (adapt version if needed):
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
imbalanced-learn
        """, language="text")
        raise e

try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop()

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ---------------------------
# Robust GET with retries
# ---------------------------
def safe_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    backoff = REQUEST_BACKOFF
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{REQUEST_RETRIES}) for {url}: {e}")
            time.sleep(backoff)
            backoff *= 2
    return None

# ---------------------------
# 1) SCRAPER with ftfy cleaning
# ---------------------------
@st.cache_data(ttl=60*60*24)  # cache for a day per arg set
def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    seen_urls = set()
    rows = []
    page_count = 0

    while current_url and page_count < MAX_PAGES:
        page_count += 1
        if current_url in seen_urls:
            logger.info("Detected repeated page, stopping to avoid infinite loop.")
            break
        seen_urls.add(current_url)

        resp = safe_get(current_url, timeout=15)
        if resp is None:
            st.warning(f"Failed to fetch {current_url} after retries; stopping scraper.")
            break

        #  Force correct text decoding
        try:
            raw_html = resp.content  # get bytes, not pre-decoded text
            decoded_html = raw_html.decode("utf-8", errors="replace")
        except Exception:
            decoded_html = resp.text  # fallback

        # Parse with BeautifulSoup using manually decoded HTML
        soup = BeautifulSoup(decoded_html, "html.parser")

        items = soup.find_all("li", class_="o-listicle__item")
        if not items:
            logger.info("No items found on page; stopping.")
            break

        stop_if_older = False
        for card in items:
            # date extraction
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(" ", strip=True) if date_div else ""
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format="%B %d, %Y")
                    except Exception:
                        claim_date = pd.to_datetime(match.group(1), errors='coerce')

            if claim_date is None:
                continue

            if claim_date < start_date:
                stop_if_older = True
                break

            if not (start_date <= claim_date <= end_date):
                continue

            # statement
            statement = None
            statement_block = card.find("div", class_="m-statement__quote")
            if statement_block:
                a = statement_block.find("a", href=True)
                if a:
                    statement = clean(a.get_text(" ", strip=True))

            # source / speaker
            source = None
            source_a = card.find("a", class_="m-statement__name")
            if source_a:
                source = clean(source_a.get_text(" ", strip=True))

            # author
            author = None
            footer = card.find("footer", class_="m-statement__footer")
            if footer:
                text = footer.get_text(" ", strip=True)
                m = re.search(r"By\s+([^•\n\r]+)", text)
                if m:
                    author = clean(m.group(1).strip())
                else:
                    parts = text.split("•")
                    if parts:
                        author = clean(parts[0].replace("By", "").strip())

            # label
            label = None
            label_img = card.find("img", alt=True)
            if label_img and 'alt' in label_img.attrs:
                label = clean(label_img['alt'].replace('-', ' ').title())

            rows.append({
                "author": author,
                "statement": statement,
                "source": source,
                "date": claim_date.strftime("%Y-%m-%d"),
                "label": label
            })

        if stop_if_older:
            break

        # next page
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and next_link.get("href"):
            current_url = urljoin(base_url, next_link['href'])
        else:
            break

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["statement", "label"])
    if not df.empty:
        df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ---------------------------
# 2) Feature functions (batch)
# ---------------------------
def lexical_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        toks = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        processed.append(" ".join(toks))
    return processed

def syntactic_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        pos = " ".join([token.pos_ for token in doc])
        processed.append(pos)
    return processed

def semantic_features_batch(texts: List[str]) -> pd.DataFrame:
    out = []
    for t in texts:
        b = TextBlob(t)
        out.append([b.sentiment.polarity, b.sentiment.subjectivity])
    return pd.DataFrame(out, columns=["polarity", "subjectivity"])

def discourse_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        sents = [sent.text.strip() for sent in doc.sents]
        first_words = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
        processed.append(f"{len(sents)} {first_words}")
    return processed

def pragmatic_features_batch(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([tl.count(w) for w in pragmatic_words])
    return pd.DataFrame(rows, columns=pragmatic_words)

# ---------------------------
# 3) Feature extraction dispatcher
# ---------------------------
def apply_feature_extraction(X_series: pd.Series, phase: str, nlp) -> Tuple[np.ndarray, Optional[object]]:
    X_texts = X_series.astype(str).tolist()
    if phase == "Lexical & Morphological":
        X_proc = lexical_features_batch(X_texts, nlp)
        # ✅ dynamically adjust min_df to avoid "max_df < min_df" error
        min_df_value = 1 if len(X_proc) < 3 else 2
        vect = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=min_df_value)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Syntactic":
        X_proc = syntactic_features_batch(X_texts, nlp)
        vect = TfidfVectorizer(max_features=5000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Semantic":
        df = semantic_features_batch(X_texts)
        return df.values, None

    if phase == "Discourse":
        X_proc = discourse_features_batch(X_texts, nlp)
        vect = CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Pragmatic":
        df = pragmatic_features_batch(X_texts)
        return df.values, None

    raise ValueError("Unknown phase")

# ---------------------------
# 4) Model helpers & evaluation
# ---------------------------
def get_models_dict():
    return {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced', probability=False)
    }

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def map_label(l):
        if pd.isna(l):
            return np.nan
        l = str(l).strip()
        if l in REAL_LABELS:
            return 1
        if l in FAKE_LABELS:
            return 0
        low = l.lower()
        if "true" in low and "mostly" not in low and "half" not in low:
            return 1
        if "false" in low or "pants" in low or "fire" in low:
            return 0
        return np.nan

    df = df.copy()
    df["target_label"] = df["label"].apply(map_label)
    return df

def evaluate_models(df: pd.DataFrame, selected_phase: str, nlp) -> pd.DataFrame:
    df = create_binary_target(df)
    df = df.dropna(subset=["target_label"])
    df = df[df["statement"].astype(str).str.len() > 10]

    X_raw = df["statement"].astype(str)
    y_raw = df["target_label"].astype(int)

    if len(np.unique(y_raw)) < 2:
        st.error("Only one class present after mapping — adjust data or date range.")
        return pd.DataFrame()

    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase, nlp)

    if isinstance(X_features_full, np.ndarray):
        X_full = X_features_full
    else:
        X_full = X_features_full

    models = get_models_dict()
    results = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_list = X_raw.tolist()

    for name, model in models.items():
        st.caption(f"Training {name}...")
        fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], []
        train_times, infer_times = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_raw)), y_raw)):
            X_train_raw = pd.Series([X_list[i] for i in train_idx])
            X_test_raw = pd.Series([X_list[i] for i in test_idx])
            y_train = y_raw.values[train_idx]
            y_test = y_raw.values[test_idx]

            if vectorizer is not None:
                if selected_phase == "Lexical & Morphological":
                    X_train_proc = lexical_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = lexical_features_batch(X_test_raw.tolist(), nlp)
                elif selected_phase == "Syntactic":
                    X_train_proc = syntactic_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = syntactic_features_batch(X_test_raw.tolist(), nlp)
                elif selected_phase == "Discourse":
                    X_train_proc = discourse_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = discourse_features_batch(X_test_raw.tolist(), nlp)
                else:
                    X_train_proc = X_train_raw.tolist()
                    X_test_proc  = X_test_raw.tolist()

                X_train = vectorizer.transform(X_train_proc)
                X_test  = vectorizer.transform(X_test_proc)
            else:
                if selected_phase == "Semantic":
                    X_train = semantic_features_batch(X_train_raw.tolist()).values
                    X_test  = semantic_features_batch(X_test_raw.tolist()).values
                elif selected_phase == "Pragmatic":
                    X_train = pragmatic_features_batch(X_train_raw.tolist()).values
                    X_test  = pragmatic_features_batch(X_test_raw.tolist()).values
                else:
                    X_train = X_train_raw.values.reshape(-1, 1)
                    X_test  = X_test_raw.values.reshape(-1, 1)

            start_train = time.time()
            try:
                if name == "Naive Bayes":
                    Xt_fit = np.abs(X_train).astype(float)
                    model.fit(Xt_fit, y_train)
                    clf = model
                else:
                    pipeline = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=3)), ("clf", model)])
                    pipeline.fit(X_train, y_train)
                    clf = pipeline

                train_time = time.time() - start_train
                start_inf = time.time()
                y_pred = clf.predict(X_test)
                infer_time = (time.time() - start_inf) * 1000.0

                fold_acc.append(accuracy_score(y_test, y_pred))
                fold_f1.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_prec.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_rec.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                train_times.append(train_time)
                infer_times.append(infer_time)
            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                fold_acc.append(0); fold_f1.append(0); fold_prec.append(0); fold_rec.append(0)
                train_times.append(0); infer_times.append(9999)

        results.append({
            "Model": name,
            "Accuracy": np.mean(fold_acc) * 100,
            "F1-Score": np.mean(fold_f1),
            "Precision": np.mean(fold_prec),
            "Recall": np.mean(fold_rec),
            "Training Time (s)": round(np.mean(train_times), 3),
            "Inference Latency (ms)": round(np.mean(infer_times), 3)
        })

    return pd.DataFrame(results)
# -----------------------------------------
# Creating a simple model prediction function
# -----------------------------------------

def ai_predict_label(statement: str, nlp) -> str:
    """Use your existing NLP + trained model to predict if statement is True/False."""
    try:
        # Use Lexical & Morphological phase as baseline
        phase = "Lexical & Morphological"

        # Check if vectorizer and model exist in session
        if "ai_vectorizer" not in st.session_state or "ai_model" not in st.session_state:
            # Train once and save
            if st.session_state.get("scraped_df", pd.DataFrame()).empty:
                st.sidebar.warning("Please scrape some data first to train the AI model.")
                return "Unknown"

            df_train = create_binary_target(st.session_state["scraped_df"])
            df_train = df_train.dropna(subset=["target_label"])

            X_train, vect = apply_feature_extraction(df_train["statement"], phase, nlp)
            y_train = df_train["target_label"]

            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(X_train, y_train)

            # Save for reuse
            st.session_state["ai_vectorizer"] = vect
            st.session_state["ai_model"] = model

        # Now use stored vectorizer + model
        vect = st.session_state["ai_vectorizer"]
        model = st.session_state["ai_model"]

        # Transform input statement using SAME vectorizer
        X_feat = vect.transform(lexical_features_batch([statement], nlp))

        # Predict
        y_pred = model.predict(X_feat)
        return "True" if y_pred[0] == 1 else "False"

    except Exception as e:
        st.sidebar.warning(f"AI prediction error: {e}")
        return "Unknown"

# ---------------------------
# Humor & critique
# ---------------------------
def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It’s concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It’s the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It’s the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It’s the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))

def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."
    df_results = df_results.copy()
    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_idx = df_results['F1-Score'].idxmax()
    best_model_row = df_results.loc[best_idx]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    headline = f"👑 The Golden Snitch Award goes to the {best_model}!"
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    return summary + roast

# ---------------------------
# STREAMLIT APP UI
# ---------------------------
def app():
    st.set_page_config(page_title='AI vs. Fact: NLP Comparator', layout='wide')
     
    # Load Google Fonts (Poppins)
    st.markdown("<link href='https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap' rel='stylesheet'>", unsafe_allow_html=True)


    st.markdown(
        """
        <style>
        .intro-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1f4068 0%, #102d4f 100%);
            border-radius: 15px;
            color: #f7f7f7;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        .intro-header h1 {
            font-size: 3.5em;
            margin-bottom: 0px;
        }
        .intro-header h3 {
            font-size: 1.5em;
            opacity: 0.8;
        }
        </style>
        <div class="intro-header">
            <h1>An AI lens that detects truth through facts.</h1>
            <h3>Unmasking Misinformation Through Intelligent Fact Verification</h3>
        </div>
        """,
        unsafe_allow_html=True)


    st.divider()

    col_left, col_center, col_right = st.columns([1, 2, 2])

    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'df_results' not in st.session_state:
        st.session_state['df_results'] = pd.DataFrame()

    # LEFT: Data Sourcing & Config
    with col_left:
        st.header("1. Data Sourcing")
        st.subheader("Politifact Time Machine 🕰️")

        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()

        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

        if st.button("Scrape Politifact Data ⛏️"):
            if start_date > end_date:
                st.error("Error: Start Date must be before or equal to End Date.")
            else:
                with st.spinner("Scraping... this may take a moment for large ranges."):
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                    if scraped_df.empty:
                        st.warning("No data scraped — try narrowing the date range or check the target site structure.")
                    else:
                        st.session_state['scraped_df'] = scraped_df
                        st.success(f"Scraping complete! {len(scraped_df)} claims harvested.")
                        st.download_button("Download scraped CSV", scraped_df.to_csv(index=False).encode('utf-8'), file_name="politifact_scraped.csv", mime="text/csv")

        st.divider()
        st.header("2. Analysis Configuration")
        phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
        selected_phase = st.selectbox("Choose the Feature Set (NLP Phase):", phases, key="selected_phase")

        if st.button("Analyze Model Showdown 🥊"):
            if st.session_state['scraped_df'].empty:
                st.error("Please scrape data first!")
            else:
                with st.spinner(f"Training models using {selected_phase} features..."):
                    df_results = evaluate_models(st.session_state['scraped_df'], selected_phase, NLP_MODEL)
                    st.session_state['df_results'] = df_results
                    st.session_state['selected_phase_run'] = selected_phase
                    if not df_results.empty:
                        st.success("Analysis complete! Results ready.")
                    else:
                        st.warning("Analysis returned no results. Check logs or data.")

    # CENTER: Metrics & visuals
    with col_center:
        st.header("3. Performance Benchmarking")
        if st.session_state['df_results'].empty:
            st.info("Awaiting model training. Configure and run the analysis in the left column.")
        else:
            df_results = st.session_state['df_results']
            st.subheader(f"Results: {st.session_state['selected_phase_run']} Features")
            st.dataframe(df_results[['Model','Accuracy','F1-Score','Training Time (s)','Inference Latency (ms)']], height=220, use_container_width=True)
            st.divider()
            st.subheader("Metric Comparison")
            metrics = ['Accuracy','F1-Score','Precision','Recall','Training Time (s)','Inference Latency (ms)']
            plot_metric = st.selectbox("Metric to Plot:", metrics, index=1, key='plot_metric_center')
            df_plot = df_results[['Model', plot_metric]].set_index('Model')
            st.bar_chart(df_plot)
            st.caption(f"Chart shows each model's mean {plot_metric} across {N_SPLITS} folds.")

            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv_data, file_name="model_results.csv", mime="text/csv")

    # ===================================================
    #  NEW SECTION: Cross-Platform Accuracy Evaluation
    # ===================================================
    st.header("5. Cross-Platform Accuracy Evaluation 🌐")

    if st.session_state.get('scraped_df', pd.DataFrame()).empty:
        st.info("Please scrape Politifact data first to perform cross-platform accuracy check.")
    else:
        num_samples = st.slider("Select number of statements to verify:", 50, len(st.session_state['scraped_df']), 500, step=50)
        df_sample = st.session_state['scraped_df'].sample(num_samples, random_state=42).reset_index(drop=True)

        if st.button("🔍 Compare with Google Fact Check"):
            st.info("Contacting Google Fact Check API... This may take a few minutes for large samples.")
            checked = []
            total = 0
            matches = 0

            for _, row in df_sample.iterrows():
                claim = str(row['statement'])
                politifact_label = normalize_label(row['label'])
                results = get_fact_check_results(claim)
                google_label = "Unknown"

                if results:
                    google_label = normalize_label(results[0].get('rating', 'Unknown'))

                checked.append({
                    "statement": claim[:150] + ("..." if len(claim) > 150 else ""),
                    "Politifact Label": politifact_label,
                    "Google Label": google_label
                })

                if google_label == politifact_label and google_label != "Unknown":
                    matches += 1
                total += 1
                time.sleep(0.4)  # prevent API rate limit

            df_compare = pd.DataFrame(checked)
            accuracy = (matches / total) * 100 if total > 0 else 0

            st.success(f"✅ Cross-platform fact-check accuracy: {accuracy:.2f}% ({matches}/{total} matched)")
            st.dataframe(df_compare, use_container_width=True)

            csv_data = df_compare.to_csv(index=False).encode('utf-8')
            st.download_button("Download Comparison CSV", csv_data, file_name="cross_platform_accuracy.csv", mime="text/csv")

    # RIGHT: critique & speed-quality plot
    with col_right:
        st.header("4. Humorous Critique")
        if st.session_state['df_results'].empty:
            st.info("The models are currently on a coffee break. Run the analysis to see results!")
        else:
            critique_text = generate_humorous_critique(st.session_state['df_results'], st.session_state['selected_phase_run'])
            st.markdown(critique_text)
            st.divider()
            st.subheader("Speed vs. Quality Trade-off")
            metrics_quality = ['Accuracy','F1-Score','Precision','Recall']
            metrics_speed = ['Training Time (s)','Inference Latency (ms)']
            x_axis = st.selectbox("X-Axis (Speed/Cost):", metrics_speed, key='x_axis', index=1)
            y_axis = st.selectbox("Y-Axis (Quality):", metrics_quality, key='y_axis', index=0)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(st.session_state['df_results'][x_axis], st.session_state['df_results'][y_axis], s=150, alpha=0.75)
            for i, row in st.session_state['df_results'].iterrows():
                ax.annotate(row['Model'], (row[x_axis] + 0.01 * st.session_state['df_results'][x_axis].max(), row[y_axis] * 0.99), fontsize=9)
            ax.set_xlabel(x_axis); ax.set_ylabel(y_axis)
            ax.set_title(f"{x_axis} vs {y_axis}")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            st.caption("Look for models in bottom-left for best balance (Low Time, High Quality).")

    # ---------------------------
    # 🔍 SIDEBAR FACT CHECK TOOL
    # ---------------------------
st.sidebar.markdown("""
    <style>
    /* Sidebar Card Styling */
    .fact-card {
        background: rgba(255,255,255,0.07);
        border-radius: 15px;
        padding: 15px 18px;
        margin-bottom: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .fact-card:hover {
        background: rgba(255,255,255,0.12);
        transform: scale(1.02);
    }

    /* Publisher Title */
    .publisher {
        color: #66d9ff;
        font-weight: 700;
        font-size: 1.05em;
    }

    /* Verdict Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.85em;
        font-weight: 600;
        margin-top: 4px;
    }
    .true {background: #00c853; color: white;}
    .false {background: #e53935; color: white;}
    .mixed {background: #ffb300; color: black;}
    .norating {background: #757575; color: white;}

    /* Link Style */
    .fact-link {
        color: #80d8ff;
        text-decoration: none;
        font-weight: 500;
    }
    .fact-link:hover {
        color: #00e5ff;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Google-fact-check.png", width=180)

# ===================================================
# Cross-Platform Fact Check
# ===================================================
st.sidebar.subheader("Cross-Platform Fact Check")

# Sidebar input field
user_query = st.sidebar.text_input("Enter a claim or statement to fact-check:")

if st.sidebar.button("Check Fact Credibility"):
    if not user_query.strip():
        st.sidebar.warning("Please enter a statement to check.")
    else:
        st.sidebar.info("Fetching verified fact-checks...")
        results = get_fact_check_results(user_query)

        if not results or (len(results) == 1 and results[0]['publisher'] == "Error"):
            st.sidebar.warning("No verified fact-checks found for this claim.")
        else:
            # AI prediction
            ai_label = ai_predict_label(user_query, NLP_MODEL)
            st.sidebar.markdown(f"""
                <div style="padding:10px; border-radius:10px; background:#1c2541; color:white; margin-bottom:10px;">
                    <b>AI Prediction:</b> {ai_label}
                </div>
            """, unsafe_allow_html=True)

            total = 0
            matches = 0

            st.sidebar.success(f"Found {len(results)} fact-check result(s):")
            for r in results[:5]:
                normalized = normalize_label(r['rating'])
                total += 1
                if normalized == ai_label:
                    matches += 1

                # Badge color selection
                if normalized == "True":
                    badge_color = "#00c851"
                    badge_icon = "✅"
                elif normalized == "False":
                    badge_color = "#ff4444"
                    badge_icon = "❌"
                elif normalized == "Mixed":
                    badge_color = "#ffbb33"
                    badge_icon = "⚠️"
                else:
                    badge_color = "#999999"
                    badge_icon = "❓"

                st.sidebar.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.08);
                    border-radius:10px;
                    padding:10px;
                    margin-bottom:8px;
                    color:white;
                ">
                    <div><b>Source:</b> {r['publisher']}</div>
                    <div><b>Verdict:</b> 
                        <span style="background:{badge_color};
                                     color:white;
                                     border-radius:6px;
                                     padding:3px 8px;
                                     font-weight:600;">
                            {badge_icon} {normalized}
                        </span>
                    </div>
                    <div style="margin-top:8px;">
                        <a href="{r['url']}" target="_blank"
                           style="color:#00c6ff; text-decoration:none; font-weight:500;">
                            View Fact Check
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Accuracy summary
            accuracy = (matches / total) * 100 if total > 0 else 0
            st.sidebar.markdown(f"""
            <hr>
            <div style="background:#132743; color:#fff;
                        border-radius:10px; padding:10px; text-align:center;">
                <b> Match Accuracy:</b> {accuracy:.2f}%
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()
