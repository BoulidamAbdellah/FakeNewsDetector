import streamlit as st
import requests
import datetime
import pickle
import tensorflow as tf
import numpy as np
import re
import joblib
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords

# Utilisation des fonctions existantes pour le traitement du texte et la prédiction
# Le code des fonctions existantes reste inchangé
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

def clean_word(word):
    """Nettoie le mot : conserve uniquement les lettres alphabétiques et met en minuscule"""
    return re.sub(r'[^a-z]', '', word.lower())

def get_combined_stopwords():
    """Combine les stopwords de toutes les sources et les nettoie"""
    # Sources de stopwords
    sources = {
        "nltk": set(nltk_stopwords.words('english')),
        "spacy": set(nlp.Defaults.stop_words),
        "sklearn": set(sklearn_stopwords),
        "gensim": set(gensim_stopwords)
    }
    
    # Combinaison et nettoyage
    combined = set()
    for source_words in sources.values():
        for word in source_words:
            cleaned = clean_word(word)  # Nettoie le mot
            if cleaned and len(cleaned):  # Ignore les mots vides ou trop courts
                combined.add(cleaned)
    
    return sorted(combined)

s_words = get_combined_stopwords()

def text_cleaning(text) : 
    global nlp, s_words
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)  # \w = alphanumérique, \s = espace
    
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)
    cleaned_text = [nlp(word)[0].lemma_ for word in text_no_punct.split() if word not in s_words]
    return " ".join(cleaned_text)

def text_to_decvect(text, model):
    vector = model.infer_vector(text.split())
    return vector   

def prepare_text(text, title):
    text = text_cleaning(text)
    title = text_cleaning(title)
    tf_idf_vectorizer = joblib.load("tf_idf_vectorizer.pkl")
    model = Doc2Vec.load("mon_modele_doc2vec.model")
    vects = tf_idf_vectorizer.transform([title, text]).toarray()
    docvects = [text_to_decvect(title, model), text_to_decvect(text, model)]
    
    simcos1 = cosine_similarity([vects[0]], [vects[1]])[0][0]
    simcos2 = cosine_similarity([docvects[0]], [docvects[1]])[0][0]
    return np.hstack([np.array([vects[0]]), np.array([vects[1]]), np.array([[simcos1, simcos2]])])

@st.cache_data
def get_claims(query, year, max_results=20):
    api_key = "AIzaSyCqaCJWmHIZtuX9hCA-k418rqZjXUB_2z8"
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    results = []
    page_token = None

    while len(results) < max_results:
        params = {
            "key": api_key,
            "query": query,
            "pageSize": min(100, max_results - len(results)),
            "languageCode": "en",
            "maxAgeDays": (datetime.datetime.now() - datetime.datetime(year, 1, 1)).days
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(base_url, params=params).json()
        if "claims" not in response:
            break

        for claim in response["claims"]:
            text = claim.get("text", "")
            reviews = claim.get("claimReview", [])
            if reviews:
                review = reviews[0]
                title = review.get("title", "")
                textual_rating = review.get("textualRating", "")  # Récupérer le rating de l'API
                publisher = review.get("publisher", {}).get("name", "")
                review_date = review.get("reviewDate", "")
                results.append({
                    "text": text,
                    "title": title,
                    "publisher": publisher,
                    "reviewDate": review_date,
                    "url": review.get("url", ""),
                    "verified": True if textual_rating else False,  # Marquer comme vérifié si un rating existe
                    "label": textual_rating,  # Utiliser le rating fourni par l'API
                    "confidence": None  # Ajout d'un champ pour stocker la confiance de la prédiction
                })

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results

def load_model(model_name):
    if model_name == "SVM":
        return joblib.load("models/svm_model.joblib")
    elif model_name == "Logistic Regression":
        return joblib.load("models/Lregression_model.joblib")
    elif model_name == "Deep Learning":
        return tf.keras.models.load_model("models/deepmodelclasifier.h5")

def predict(model, model_name, text, title):
    preprocessed = prepare_text(text, title)
    if model_name == "Deep Learning":
        return model.predict((preprocessed))[0][0]
    else:
        print(model.predict((preprocessed))[0])
        return model.predict(preprocessed)[0]

# ---------------- NOUVELLE INTERFACE STREAMLIT ----------------

st.set_page_config(page_title="Vérificateur de Fake News", layout="wide")

# Styles CSS personnalisés pour les cartes et éléments de l'interface
st.markdown("""
<style>
    .news-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        min-height: 200px; /* Hauteur minimale pour les cartes */
    }
    .card-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1e3a8a;
    }
    .card-source {
        color: #6c757d;
        margin-bottom: 10px;
        font-style: italic;
    }
    .card-date {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 15px;
    }
    .card-content {
        margin: 15px 0;
        line-height: 1.5;
        color: #333;
    }
    .card-link {
        display: inline-block;
        margin-top: 10px;
        color: #0d6efd;
        text-decoration: none;
        font-weight: 500;
    }
    .card-link:hover {
        text-decoration: underline;
    }
    .verification-badge {
        position: absolute;
        top: 15px;
        right: 15px;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 14px;
    }
    .real-badge {
        background-color: #28a745;
        color: white;
    }
    .fake-badge {
        background-color: #dc3545;
        color: white;
    }
    .unverified-badge {
        background-color: #ffc107;
        color: black;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
    }
    .page-title {
        color: #4682b4;
        margin-bottom: 0;
    }
    /* Amélioration pour les boutons */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4682b4;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a6d99;
    }
</style>
""", unsafe_allow_html=True)

# En-tête personnalisé
# Titre principal
st.markdown(
    '<div class="header-container">'
    '<h1 class="page-title">📰 Vérificateur de Fake News</h1>'
    '<p>TF-IDF + Doc2Vec</p>'
    '</div>', unsafe_allow_html=True
)

# Conteneur pour les filtres de recherche
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("🔍 Mot-clé pour chercher des news :", "climate change")
    with col2:
        year = st.number_input("📅 Année :", min_value=2000, max_value=2025, value=2017)
    with col3:
        max_news = st.slider("🔢 Nombre de résultats :", 1, 50, 10)

# Choix du modèle
model_choice = st.selectbox("🧠 Modèle de vérification :", ["SVM", "Logistic Regression", "Deep Learning"])
model = None

# Recherche des nouvelles
if st.button("🔎 Rechercher"):
    with st.spinner("Recherche en cours..."):
        news = get_claims(query, year, max_news)
        if not news:
            st.warning("❌ Aucune nouvelle trouvée.")
        else:
            st.session_state.news = news
            st.session_state.model_choice = model_choice
            with st.spinner("Chargement du modèle..."):
                model = load_model(model_choice)
                st.session_state.model = model
            st.success(f"✅ {len(news)} nouvelles trouvées.")

# Affichage des cartes
if 'news' in st.session_state:
    news = st.session_state.news
    model_choice = st.session_state.model_choice
    model = st.session_state.model
    
    st.markdown("### 📰 Résultats de recherche")
    cols = st.columns(2)  # Deux cartes par ligne

    for i, item in enumerate(news):
        col_index = i % 2
        
        # Préparer les données
        title = item.get("title", "Sans titre")
        text = item.get("text", "")
        publisher = item.get("publisher", "Source inconnue")
        review_date = item.get("reviewDate", "Date inconnue")
        url = item.get("url", "#")
        verified = item.get("verified", False)
        label = item.get("label", "")
        confidence = item.get("confidence", None)

        # Badge de statut initial
        badge_class = "unverified-badge"
        badge_text = "Non vérifié"
        if verified and label:
            if "false" in label.lower() or "fake" in label.lower() or "faux" in label.lower():
                badge_class = "fake-badge"
                badge_text = "FAKE"
            elif "true" in label.lower() or "real" in label.lower() or "vrai" in label.lower():
                badge_class = "real-badge"
                badge_text = "REAL"
            else:
                badge_text = label

        # Limiter le texte affiché
        display_text = text[:200] + "..." if text and len(text) > 200 else text

        # Carte HTML
        card_html = f"""
        <div class="news-card">
            <div class="verification-badge {badge_class}">{badge_text}</div>
            <div class="card-title">{title}</div>
            <div class="card-source">Source: {publisher}</div>
            <div class="card-date">Date: {review_date}</div>
            <div class="card-content">{display_text}</div>
            <a href="{url}" target="_blank" class="card-link">Lire l'article original</a>
        </div>
        """

        with cols[col_index]:
            st.markdown(card_html, unsafe_allow_html=True)

            # Bouton de vérification
            if st.button(f"🧪 Vérifier cette nouvelle", key=f"verify_{i}"):
                with st.spinner("Analyse en cours..."):
                    prediction = predict(model, model_choice, text, title)

                    if model_choice == "Deep Learning":
                        label = "FAKE" if prediction <= 0.5 else "REAL"
                        confidence = prediction if prediction >= 0.5 else 1 - prediction
                    else:
                        label = "FAKE" if prediction == 0 else "REAL"
                        confidence = 1.0  # SVM/LR sans probas

                    # Mise à jour
                    st.session_state.news[i]["verified"] = True
                    st.session_state.news[i]["label"] = label
                    st.session_state.news[i]["confidence"] = confidence

                    st.rerun()

            # Affichage du badge de résultat si déjà vérifié
            if verified:
                confidence_text = f"{int(confidence * 100)}%" if confidence is not None else "N/A"
                result_class = "real-badge" if label == "REAL" else "fake-badge"
                st.markdown(f"""
                <div class="verification-badge {result_class}" style="position: static; margin-top: 10px;">
                    {label} (Confiance: {confidence_text})
                </div>
                """, unsafe_allow_html=True)
