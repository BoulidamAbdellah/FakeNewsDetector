# Détection de Fausses Nouvelles : Performances Comparées de Modèles d'Apprentissage Automatique

## 📖 À Propos du Projet

Ce projet vise à développer et évaluer un système de détection de fausses nouvelles (fake news) basé sur des techniques d'apprentissage automatique et de traitement du langage naturel (NLP). L'objectif principal est de comparer les performances de différents modèles de classification pour distinguer les articles de presse authentiques des articles fallacieux, en se basant sur leur contenu textuel (titre et corps de l'article).

Le pipeline complet du projet comprend :
1.  **Collecte et Prétraitement des Données** : Utilisation de jeux de données publics contenant des nouvelles étiquetées "vraies" et "fausses".
2.  **Nettoyage Approfondi du Texte** : Normalisation (minuscules, suppression de la ponctuation), suppression d'une liste exhaustive de mots vides (stopwords) et lemmatisation.
3.  **Ingénierie des Caractéristiques (Feature Engineering)** :
    *   Extraction de caractéristiques **TF-IDF** (Term Frequency-Inverse Document Frequency) en considérant les unigrammes et bigrammes.
    *   Apprentissage de plongements de documents avec **Doc2Vec**.
    *   Calcul de la **similarité cosinus** entre les représentations vectorielles des titres et des corps de texte (pour TF-IDF et Doc2Vec).
    *   Combinaison de ces éléments pour former des ensembles de caractéristiques robustes.
4.  **Entraînement et Évaluation de Modèles** :
    *   Régression Logistique
    *   Machine à Vecteurs de Support (SVM)
    *   Réseau de Neurones Profonds (avec Keras/TensorFlow)
5.  **Application Web Interactive** : Développement d'une interface utilisateur avec Streamlit pour permettre la recherche d'articles et leur vérification à la demande à l'aide des modèles entraînés.

Ce dépôt contient le code source des notebooks Jupyter pour le traitement des données et l'entraînement des modèles, ainsi que le code de l'application Streamlit.

## 🚀 Fonctionnalités Clés

*   **Pipeline de Prétraitement Textuel Complet** : Du nettoyage de base à la lemmatisation avancée.
*   **Ingénierie de Caractéristiques Multiples** : Combine la puissance de TF-IDF et la sémantique de Doc2Vec.
*   **Comparaison de Modèles de Classification** : Évaluation rigoureuse de plusieurs algorithmes d'apprentissage automatique.
*   **Application Streamlit Interactive** :
    *   Recherche d'articles de presse via l'API Google Fact Check Tools.
    *   Sélection du modèle de classification (SVM, Régression Logistique, Deep Learning).
    *   Vérification à la demande de la véracité des articles.
    *   Affichage clair des prédictions et (le cas échéant) des scores de confiance.

## 🛠️ Technologies et Bibliothèques Utilisées

*   **Langage** : Python 3
*   **Manipulation de Données** : Pandas, NumPy
*   **Traitement du Langage Naturel (NLP)** :
    *   NLTK (stopwords, tokenisation)
    *   spaCy (lemmatisation, traitement efficace du texte)
    *   Gensim (Doc2Vec)
    *   Scikit-learn (TfidfVectorizer, gestion des stopwords)
*   **Apprentissage Automatique** :
    *   Scikit-learn (LogisticRegression, SVC, métriques d'évaluation, train_test_split)
    *   TensorFlow / Keras (modèles de réseaux de neurones profonds)
*   **Application Web** : Streamlit
*   **Utilitaires** : Joblib (sauvegarde/chargement de modèles), tqdm (barres de progression), Matplotlib & Seaborn (visualisations)
*   **API** : Google Fact Check Tools API (via la bibliothèque `requests`)

## 📂 Structure du Dépôt

```
.
├── data/                           # (Optionnel) Peut contenir les jeux de données bruts (Fake.csv, True.csv) - Attention à la taille !
├── notebooks/                      # Contient les notebooks Jupyter
│   ├── 1_data_cleaning_preprocessing.ipynb # Nettoyage, prétraitement, ingénierie des caractéristiques
│   └── 2_model_training_evaluation.ipynb # Entraînement et évaluation des modèles
├── models/                         # Modèles entraînés sauvegardés (.joblib, .h5)
│   ├── Lregression_model.joblib
│   ├── svm_model.joblib
│   └── deepmodelclassifier.h5
├── app.py                          # Code source de l'application Streamlit
├── requirements.txt                # Dépendances Python nécessaires
├── README.md                       # Ce fichier
└── .gitignore                      # Fichiers et dossiers à ignorer par Git (ex: __pycache__, venv)
```
*(Note : Vous devrez peut-être ajuster cette structure pour correspondre exactement à votre dépôt.)*

## ⚙️ Installation et Configuration

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/VOTRE_NOM_UTILISATEUR/NOM_DE_VOTRE_DEPOT.git
    cd NOM_DE_VOTRE_DEPOT
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Activer l'environnement :
    # Sur Windows :
    # .\venv\Scripts\activate
    # Sur macOS/Linux :
    # source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Télécharger les ressources NLTK et spaCy :**
    Exécutez les commandes suivantes dans un interpréteur Python ou au début de vos notebooks :
    ```python
    import nltk
    import spacy

    nltk.download('stopwords')
    spacy.cli.download('en_core_web_sm')
    ```

5.  **Jeux de Données :**
    *   Placez les fichiers `Fake.csv` et `True.csv` dans un répertoire approprié (par exemple, un dossier `data/` à la racine du projet) si vous ne les incluez pas directement dans le suivi Git en raison de leur taille. Mettez à jour les chemins dans les notebooks si nécessaire.
    *   *Alternativement, fournissez un lien de téléchargement ou des instructions pour obtenir les jeux de données.*

6.  **Clé API Google Fact Check Tools (pour l'application Streamlit) :**
    *   L'application Streamlit utilise l'API Google Fact Check Tools. Vous devrez obtenir votre propre clé API.
    *   La clé est actuellement codée en dur dans le script `app.py` (`api_key = "AIzaSyCqaCJWmHIZtuX9hCA-k418rqZjXUB_2z8"`). **Il est fortement recommandé de ne PAS laisser de clés API en clair dans le code sur un dépôt public.**
    *   Utilisez plutôt des variables d'environnement ou un fichier de configuration (ignoré par Git) pour gérer votre clé API. Par exemple, avec Streamlit Secrets pour les applications déployées.

## 🚀 Exécution

### 1. Notebooks Jupyter
Ouvrez et exécutez les notebooks dans l'ordre :
1.  `notebooks/1_data_cleaning_preprocessing.ipynb` : Cette étape générera les fichiers de caractéristiques (`.npz`) et les modèles de vectorisation (`.pkl`) nécessaires pour l'étape suivante. Assurez-vous que les fichiers de données d'entrée (`Fake.csv`, `True.csv`) sont au bon emplacement.
2.  `notebooks/2_model_training_evaluation.ipynb` : Ce notebook charge les caractéristiques prétraitées, entraîne les différents modèles de classification, les évalue et sauvegarde les modèles entraînés dans le dossier `models/`.

### 2. Application Streamlit
Une fois les modèles entraînés et sauvegardés dans le dossier `models/` (et les fichiers `tf_idf_vectorizer.pkl` et `mon_modele_doc2vec.model` disponibles à la racine ou dans un chemin accessible), vous pouvez lancer l'application Streamlit :
```bash
streamlit run app.py
```
Ouvrez votre navigateur web à l'adresse locale fournie (généralement `http://localhost:8501`).

## 📊 Résultats Attendus (Exemple)

Les modèles de Régression Logistique et d'Apprentissage Profond ont montré des performances exceptionnelles sur les jeux de données utilisés, avec des précisions (accuracy) supérieures à 99%. Le modèle SVM a également bien performé, bien que légèrement en retrait, potentiellement en raison d'une configuration non exhaustive des hyperparamètres.

Consultez le rapport complet du projet (si disponible/lié) pour une analyse détaillée des performances et des visualisations.

## 🤝 Contribution

Les contributions, suggestions et rapports de bugs sont les bienvenus ! N'hésitez pas à ouvrir une "issue" ou à proposer une "pull request".

## 📜 Licence

Ce projet est sous licence [NOM_DE_LA_LICENCE - ex: MIT License]. Voir le fichier `LICENSE` (si vous en ajoutez un) pour plus de détails.

## 🙏 Remerciements (Optionnel)

*   Remerciements à [Nom de l'encadrant(e)] pour ses conseils et son soutien.
*   Sources des jeux de données (si différentes de celles implicites).

---
*Dernière mise à jour : [Date]*
```

---

## Proposition pour `requirements.txt`

```txt
pandas
numpy
scikit-learn
joblib
nltk
spacy
gensim
tqdm
tensorflow # Ou tensorflow-cpu si vous n'avez pas besoin du GPU
streamlit
requests
matplotlib # Si vous générez des graphiques dans les notebooks et voulez fixer la version
seaborn    # Si vous générez des graphiques dans les notebooks et voulez fixer la version

# Pour le téléchargement du modèle spaCy via pip (optionnel, mais peut être utile)
# spacy[en_core_web_sm]
```

**Explications et points importants pour le `requirements.txt` :**

*   **Versions :** Pour une meilleure reproductibilité, il est fortement recommandé de spécifier les versions exactes des bibliothèques. Vous pouvez générer cela automatiquement après avoir configuré votre environnement et installé tout ce dont vous avez besoin :
    ```bash
    pip freeze > requirements.txt
    ```
    Cela listera toutes les bibliothèques de votre environnement avec leurs versions exactes. Vous pourriez ensuite nettoyer ce fichier pour ne garder que les dépendances directes de votre projet si vous le souhaitez, mais inclure toutes les dépendances figées est plus sûr pour la reproductibilité.
*   **TensorFlow :** J'ai mis `tensorflow`. Si votre projet n'utilise pas de GPU ou si vous voulez une installation plus légère, vous pouvez spécifier `tensorflow-cpu`.
*   **Matplotlib/Seaborn :** Inclus au cas où vous les utiliseriez directement dans les notebooks pour des visualisations qui ne sont pas couvertes par les fonctions de plot du code d'entraînement.
*   **`spacy[en_core_web_sm]` :** C'est une manière de s'assurer que le modèle `en_core_web_sm` est téléchargé lors de l'installation via pip, mais la méthode `spacy.cli.download('en_core_web_sm')` reste souvent plus explicite.

---

**Conseils supplémentaires pour votre dépôt GitHub :**

1.  **`.gitignore` :** Assurez-vous d'avoir un fichier `.gitignore` pour exclure les fichiers inutiles ou sensibles (comme les environnements virtuels `venv/`, les caches Python `__pycache__/`, les fichiers de configuration locaux avec des clés API, les gros fichiers de données si vous ne voulez pas les versionner).
2.  **Licence :** Si c'est un projet public, ajoutez un fichier `LICENSE` (par exemple, MIT, Apache 2.0).
3.  **Chemins de fichiers :** Vérifiez que tous les chemins de fichiers dans votre code (notebooks, `app.py`) sont relatifs à la racine du projet ou gérés de manière à ce que d'autres utilisateurs puissent exécuter le code sans modifications majeures.
4.  **Taille des données :** Les fichiers CSV peuvent être volumineux. GitHub a des limites de taille pour les fichiers et les dépôts. Si vos fichiers de données sont trop gros, envisagez de les héberger ailleurs (par exemple, Google Drive, Kaggle Datasets, AWS S3) et de fournir des instructions de téléchargement dans le README.
5.  **Clés API :** **Répétez : Ne committez jamais de clés API directement dans votre code sur un dépôt public.** Utilisez des variables d'environnement, Streamlit Secrets, ou des fichiers de configuration locaux (ajoutés au `.gitignore`).

J'espère que cela vous aidera à créer un excellent dépôt GitHub pour votre projet !
