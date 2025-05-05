import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import textwrap
import zipfile
import tempfile
import os
from collections import Counter
import openai
from openai import OpenAI
import json


# Initialisation du modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Clé API NewsAPI
API_KEY = 'a5a5a71c72b74b2a9e09e9ed9af542ca'
HUNTER_API_KEY = "2222faabcb33adf838c9f77752b5451af39ce567" 
GPT_KEY = "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"
ALPHA_API_KEY = "J1Q1MOLI77ZX9T6F"
client = OpenAI(api_key = GPT_KEY) 


def score_business_model(business_model):
    if pd.isna(business_model):
        return 0
    bm = business_model.lower()
    if "b2b" in bm and "b2c" in bm:
        return 1
    elif "b2b" in bm:
        return 2
    elif "b2c" in bm:
        return 0
    return 0


# Liste des technologies agiles
tech_agiles = [
    "docker", "kubernetes", "microservices", "ci/cd", "devops",
    "agile", "scrum", "kanban", "cloud", "aws", "gcp", "azure",
    "react", "vue", "typescript", "fastapi", "graphql", "terraform"
]

# Fonctions de scoring
def score_location(location):
    if pd.isna(location): return 0
    location = location.lower()
    if "france" in location: return 3
    elif "united states" in location or "usa" in location: return 3
    elif "europe" in location: return 2
    elif "india" in location or "china" in location: return 1
    return 0

def score_headcount(headcount):
    if headcount == "5001-10000": return 3
    elif headcount in ["10001+", "1001-5000"]: return 2
    elif headcount in ["51-200", "201-500"]: return 1
    return 0

def score_industry(industry):
    industry = industry.lower() if pd.notna(industry) else ""
    if "biotechnology research" in industry: return 3
    elif "pharmaceutical" in industry or "healthcare" in industry: return 2
    return 0

def score_company_type(company_type):
    if company_type == "privately held": return 3
    elif company_type == "public company": return 0
    return 1

def score_technologies(techs):
    if pd.isna(techs): return 0
    techs = techs.lower()
    score = sum(1 for tech in tech_agiles if tech in techs)
    return min(score, 3)

# Fonction de calcul de scores
def compute_scores(df):
    scores = []
    for _, row in df.iterrows():
        location_score = score_location(row["Location"])
        headcount_score = score_headcount(row["Headcount"])
        industry_score = score_industry(row["Industry"])
        company_type_score = score_company_type(row["Company Type"])
        tech_score = score_technologies(row["Technologies"])
        business_score = score_business_model(row["Business Model"])
        
        total_score = location_score + headcount_score + industry_score + company_type_score + tech_score + business_score
        scores.append([location_score, headcount_score, industry_score, company_type_score, tech_score, business_score, total_score])

    df["Location Score"] = [score[0] for score in scores]
    df["Headcount Score"] = [score[1] for score in scores]
    df["Industry Score"] = [score[2] for score in scores]
    df["Company Type Score"] = [score[3] for score in scores]
    df["Tech Score"] = [score[4] for score in scores]
    df["Business Score"] = [score[5] for score in scores]
    df["Total Score"] = [score[6] for score in scores]  # Ajouter la colonne Total Score
    
    return df

# Fonction pour tracer un graphique radar
def plot_radar_chart(row):
    # Les catégories
    categories = ["Location", "Headcount", "Industry", "Company Type", "Tech", "Business"]
    
    # Extraire les valeurs des scores de chaque catégorie
    values = [
        row["Location Score"],
        row["Headcount Score"],
        row["Industry Score"],
        row["Company Type Score"],
        row["Tech Score"],
        row["Business Score"]
    ]
    
    # Ajouter la première valeur pour fermer le graphique (afin d'avoir un graphique circulaire)
    values.append(values[0])  # Fermeture du cercle
    
    # Calcul des angles pour chaque catégorie
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])  # Ajouter l'angle du premier score à la fin pour fermer le cercle
    
    # Création de la figure et de l'axe polaire
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Tracer la ligne et remplir l'aire
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='b')
    ax.fill(angles, values, alpha=0.25, color='b')
    
    # Ajouter des labels et un titre
    ax.set_xticks(angles[:-1])  # Retirer le dernier angle qui est une doublure
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title(f"📊 Score Radar de {row['Company Name']}", fontsize=10)
    ax.set_yticklabels([])  # Pas de labels sur l'axe Y
    
    return fig


def show_multiple_radar_charts(df, max_companies=3):
    st.markdown("### Comparaison des scores (graphiques radars)")
    
    # Limite le nombre d'entreprises à afficher
    df_subset = df.head(max_companies)
    
    cols = st.columns(len(df_subset))  # Création de colonnes horizontales

    for i, row in enumerate(df_subset.itertuples()):
        with cols[i]:
            fig = plot_radar_chart(row) 
            st.pyplot(fig)
            st.markdown(f"**{row._asdict().get('Company Name', f'Entreprise {i+1}')}**")


# Fonction pour récupérer les news
def get_news(company_name):
    today = datetime.today().strftime('%Y-%m-%d')
    seven_days_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q={company_name}&from={seven_days_ago}&to={today}&sortBy=publishedAt&apiKey={API_KEY}'
    try:

        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return articles[:10]
    except:
        return []

# Fonction pour détecter des features
def detect_features_from_text(text, url):
    found = {}
    text = text.lower()
    if "ceo" in text and any(w in text for w in ["appointed", "named", "joins", "new ceo", "new job"]):
        # Retourner le lien sous la forme "link" en bleu
        found['Nouveau Directeur'] = (text, f"[link]({url})")
    if any(w in text for w in ["raised", "funding", "secured"]) and "$" in text:
        found['Levee de Fonds'] = (text, f"[link]({url})")
    if any(w in text for w in ["acquired", "acquisition", "merger", "merge"]):
        found['Acquisition/Fusion'] = (text, f"[link]({url})")
    if any(w in text for w in ["expanding", "expansion", "new office", "opens in"]):
        found['Expansion Géographique'] = (text, f"[link]({url})")
    if any(w in text for w in ["launched", "launches", "new product", "unveils"]):
        found['Lancement Produit'] = (text, f"[link]({url})")
    return found

def get_ticker(company_name, alpha_api_key):
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={alpha_api_key}"
    try:
        response = requests.get(url)
        matches = response.json().get('bestMatches', [])
        if matches:
            return matches[0].get('1. symbol', None)
    except Exception as e:
        print(f"❌ Erreur lors de la recherche de ticker pour {company_name}: {e}")
    return None

# Fonction pour récupérer le sentiment via Alpha Vantage
def get_sentiment_for_ticker(ticker, alpha_api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'feed' in data and data['feed']:
            headlines = [item['title'] for item in data['feed'][:3]]  # Obtenez les titres des 3 premières nouvelles
            return " | ".join(headlines)  # Renvoyer les titres comme sentiment
    except Exception as e:
        print(f"⚠️ Erreur sentiment pour {ticker}: {e}")
    return "NaN"


# Fonction pour enrichir avec les actualités, les chiffres d'affaires 2024 et le sentiment
def enrich_with_news_and_revenue(df, news_api_key, revenue_api_key, alpha_api_key):
    # Enrichir avec les actualités
    features = ['Nouveau Directeur', 'Levee de Fonds', 'Acquisition/Fusion', 'Expansion Géographique', 'Lancement Produit']
    for f in features:
        df[f] = 'NaN'
    df['News Score'] = 0
    df['Sentiment'] = 'NaN'  # Ajouter la colonne Sentiment

    for idx, row in df.iterrows():
        name = row['Company Name']
        news_items = get_news(name)  # Fonction que tu as déjà définie
        news_score = 0
        for article in news_items:
            title = article.get('title', '')
            desc = article.get('description', '')
            content = f"{title} {desc}"
            url = article.get('url', '')
            detected = detect_features_from_text(content, url)  # Fonction que tu as déjà définie
            for feature, (text, link) in detected.items():
                df.at[idx, feature] = f"{text[:100]}... | {link}"
                news_score += 1
        df.at[idx, 'News Score'] = news_score

        # Récupérer le ticker pour chaque entreprise et obtenir le sentiment
        ticker = get_ticker(name, alpha_api_key)  # Fonction déjà définie pour obtenir le ticker
        if ticker:
            sentiment = get_sentiment_for_ticker(ticker, alpha_api_key)  # Appel à Alpha Vantage
            df.at[idx, 'Sentiment'] = sentiment
        else:
            df.at[idx, 'Sentiment'] = "Ticker non trouvé"

        time.sleep(1)  # Respect du rate limit de l'API NewsAPI et Alpha Vantage

    # Enrichir avec les chiffres d'affaires 2024
    companies = df['Company Name'].dropna().tolist()

    # Prompt GPT pour récupérer les chiffres d'affaires
    prompt = (
        f"Voici une liste de {len(companies)} entreprises, chacune sur une nouvelle ligne :\n"
        + "\n".join(companies)
        + "\n\nPour chaque entreprise, donne uniquement une estimation de son chiffre d'affaires pour l'année 2024 en dollars américains (USD). "
        "Si l'information exacte est disponible, donne l'estimation la plus précise possible. "
        "Si l'information exacte n'est pas disponible, donne une estimation raisonnable basée sur la notoriété, la taille, et le secteur de l'entreprise. "
        "La réponse doit contenir uniquement des chiffres bruts (ex: 1200000000 pour 1.2 milliard), sans texte, symbole $, ni unité. "
        "Si tu n'as pas d'estimation ou de données disponibles, retourne une estimation du CA en 2024 de l'entreprise sachant toute les infos que tu as. "
        "Retourne exactement une ligne par entreprise, sans texte supplémentaire, et sans unité. "
        "Chaque ligne doit être un nombre entier, sans décimales."
    )


    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {revenue_api_key}"
        }
        data = {
            "model": "gpt-4",  # Ou "gpt-3.5-turbo"
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            # Affichage de la réponse brute pour déboguer
            content = response.json()['choices'][0]['message']['content']
            st.write(f"Réponse de l'API : {content}")  # Affiche la réponse brute ici pour déboguer

            # Diviser la réponse en lignes (une par entreprise)
            revenues = content.strip().split('\n')
            st.write(f"Revenus après découpe : {revenues}")  # Affiche les revenus après découpe pour déboguer

            # Vérifier la longueur de la réponse et la comparer à celle des entreprises
            if len(revenues) == len(companies):
                revenues = [revenue.strip() if revenue.strip() != "NaN" else None for revenue in revenues]
                df['2024 Revenue (USD)'] = pd.Series(revenues)
            else:
                raise Exception("Mauvaise longueur de réponse de l’API.")
        else:
            raise Exception(f"Erreur API OpenAI : {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"❌ Erreur enrichissement chiffre d'affaires : {e}")

    return df



# Fonction pour récupérer les contacts via Hunter.io
def get_top_1000_contacts(df, api_key = HUNTER_API_KEY, delay_between_calls=1.5):
    # Trier les entreprises selon le score et garder les 1000 premières
    top_1000 = df.sort_values(by="Total Score", ascending=False).head(1000)
    all_contacts = []

    for i, company_name in enumerate(top_1000["Company Name"], start=1):
        try:
            print(f"🔍 [{i}/1000] Recherche de contacts pour : {company_name}")
            domain_search_url = f"https://api.hunter.io/v2/domain-search?company={company_name}&api_key={api_key}"
            resp = requests.get(domain_search_url)
            if resp.status_code != 200:
                print(f"⚠️ Erreur avec {company_name} : {resp.status_code}")
                continue
            data = resp.json()
            emails = data.get("data", {}).get("emails", [])
            domain = data.get("data", {}).get("domain", "")
            for email in emails:
                all_contacts.append({
                    "Company": company_name,
                    "Email": email.get("value", ""),
                    "First Name": email.get("first_name", ""),
                    "Last Name": email.get("last_name", ""),
                    "Position": email.get("position", ""),
                    "Domain": domain
                })
            time.sleep(delay_between_calls)  # Délai pour éviter d'être bloqué par l'API
        except Exception as e:
            print(f"❌ Erreur avec {company_name} : {str(e)}")
            continue

    print(f"✅ Terminé ! {len(all_contacts)} contacts récupérés.")
    return all_contacts

    
    

def enrich_business_model_column(df, api_key):
    if "Business Model" in df.columns:
        return df  # déjà présent

    companies = df['Company Name'].dropna().tolist()

    prompt = (
        f"Voici une liste de {len(companies)} entreprises, chacune sur une nouvelle ligne :\n"
        + "\n".join(companies)
        + "\n\nJe veux que tu me renvoies une liste de la même longueur, "
        "avec exactement une valeur par ligne. Les valeurs possibles sont : B2B, B2C ou B2B et B2C. "
        "Chaque valeur doit correspondre au modèle d'affaires de l'entreprise respective. "
        f"Assure-toi que la réponse contient exactement {len(companies)} lignes, "
        "une pour chaque entreprise, sans ajouter de texte supplémentaire."
    )

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            business_models = content.strip().split('\n')
            if len(business_models) == len(companies):
                business_models = [model.strip() for model in business_models]
                df['Business Model'] = pd.Series(business_models)
                return df
            else:
                raise Exception("Mauvaise longueur de réponse de l’API.")
        else:
            raise Exception(f"Erreur API OpenAI : {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"❌ Erreur enrichissement modèle d'affaires : {e}")
        return df


# Fonction pour afficher la fiche technique d'une entreprise
def show_company_profile(company_df):
    # On suppose que company_df est une DataFrame qui contient les informations d'une entreprise.
    row = company_df.iloc[0]
    
    st.markdown(f"### Fiche technique de {row['Company Name']}")

    # Afficher le graphique radar
    fig = plot_radar_chart(row)
    st.pyplot(fig)
    
    # Afficher le chiffre d'affaires
    st.markdown(f"**Chiffre d'affaires 2024 :** {row['2024 Revenue (USD)']}")
    
    # Afficher le sentiment
    st.markdown(f"**Sentiment :** {row['Sentiment']}")
    
    # Afficher les contacts
    if "contacts_df" in st.session_state:
        contacts_for_company = st.session_state["contacts_df"][st.session_state["contacts_df"]["Company"] == row["Company Name"]]
        if not contacts_for_company.empty:
            st.markdown("### Contacts")
            st.dataframe(contacts_for_company[["First Name", "Last Name", "Position", "Email"]])
        else:
            st.markdown("Aucun contact trouvé pour cette entreprise.")
    
    # Afficher les actualités
    if row["News Score"] > 0:
        st.markdown(f"### Actualités ({row['News Score']} nouvelles)")
        for feature in ['Nouveau Directeur', 'Levee de Fonds', 'Acquisition/Fusion', 'Expansion Géographique', 'Lancement Produit']:
            if pd.notna(row[feature]):
                st.markdown(f"- **{feature}:** {row[feature]}")
    else:
        st.markdown("Aucune actualité récente trouvée.")

    # Afficher les informations supplémentaires
    st.markdown(f"**Localisation :** {row['Location']}")
    st.markdown(f"**Taille de l'entreprise :** {row['Headcount']}")
    st.markdown(f"**Secteur :** {row['Industry']}")
    st.markdown(f"**Type d'entreprise :** {row['Company Type']}")

    # Ajouter un bouton pour revenir à la liste des entreprises
    if st.button("🔙 Retour à la liste des entreprises"):
        st.session_state["enriched_df"]  # Redonne accès au DataFrame enrichi
        st.experimental_rerun()

def generate_company_pdf_zip(df_top5):
    # Créer un répertoire temporaire
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = os.path.join(temp_dir.name, "fiches_entreprises.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i, row in df_top5.iterrows():
            # Générer le PDF individuel
            pdf_buffer = generate_company_pdf(row)  # Ta fonction personnalisée pour chaque fiche

            # Ajouter au ZIP
            pdf_filename = f"{row['Company Name'].replace('/', '_')}_fiche.pdf"
            zipf.writestr(pdf_filename, pdf_buffer.getvalue())

    return zip_path, temp_dir

def generate_company_pdf(row):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    # === Titre ===
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"📄 Fiche Technique - {row['Company Name']}")
    y -= 30

    # === Infos générales ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "1. Informations générales")
    y -= 18

    c.setFont("Helvetica", 10)
    infos = [
        f"📍 Localisation : {row.get('Location', 'N/A')}",
        f"🏭 Secteur : {row.get('Industry', 'N/A')}",
        f"🏢 Type d'entreprise : {row.get('Company Type', 'N/A')}",
        f"👥 Taille (Headcount) : {row.get('Headcount', 'N/A')}",
        f"💼 Modèle économique : {row.get('Business Model', 'N/A')}",
    ]
    for info in infos:
        c.drawString(margin + 10, y, info)
        y -= 14

    y -= 10

    # === Scores ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "2. Scores de l'entreprise")
    y -= 18

    c.setFont("Helvetica", 10)
    scores = [
        f"⭐ Score Total : {row.get('Total Score', 0):.2f}",
        f"🔧 Tech Score : {row.get('Tech Score', 0):.2f}",
        f"📈 Business Score : {row.get('Business Score', 0):.2f}",
        f"📍 Location Score : {row.get('Location Score', 0):.2f}",
        f"🏢 Company Type Score : {row.get('Company Type Score', 0):.2f}",
        f"👥 Headcount Score : {row.get('Headcount Score', 0):.2f}",
        f"🏭 Industry Score : {row.get('Industry Score', 0):.2f}",
    ]
    for score in scores:
        c.drawString(margin + 10, y, score)
        y -= 14

    y -= 10

    # === Sentiment & News ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "3. Actualités et sentiment")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(margin + 10, y, f"📰 News Score : {row.get('News Score', 'N/A')}")
    y -= 14
    c.drawString(margin + 10, y, f"💬 Sentiment : {row.get('Sentiment', 'N/A')}")
    y -= 30

    # === Graphique Radar ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "4. Graphique radar des scores")
    y -= 150  # Réserver la hauteur pour le graphique

    fig = plot_radar_chart(row)
    img_stream = io.BytesIO()
    fig.savefig(img_stream, format="PNG", dpi=150)
    plt.close(fig)
    img_stream.seek(0)
    c.drawImage(ImageReader(img_stream), margin, y, width=250, height=250)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer



def generate_statistical_summary(df):
    st.header("📊 Fiche Statistique - Top 50 entreprises")
    
    st.subheader("1. Distribution des variables numériques")
    numerical_cols = ["Total Score", "Tech Score", "Location Score", "Company Type Score", "Headcount Score"]
    
    for col in numerical_cols:
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    st.subheader("2. Répartition des variables catégorielles")
    categorical_cols = ["Location", "Industry", "Business Model", "Headcount", "Company Type"]
    
    for col in categorical_cols:
        st.markdown(f"**{col}**")
        value_counts = df[col].value_counts(normalize=True) * 100
        for category, pct in value_counts.items():
            st.markdown(f"- {category}: {pct:.1f}%")



def generate_pdf(df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 50
    col_gap = 30
    col_width = (width - 2 * margin - col_gap) / 2
    y_position = height - margin

    # Titre principal
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y_position, "📊 Fiche Statistique - Top 50 Entreprises")
    y_position -= 25

    # Phrase d’introduction (multi-lignes auto)
    intro_text = "Cette fiche regroupe les statistiques de vos potentiels clients. Elle est essentielle pour guider vos décisions stratégiques."
    wrapped_intro = textwrap.wrap(intro_text, width=95)
    c.setFont("Helvetica", 11)
    for line in wrapped_intro:
        c.drawString(margin, y_position, line)
        y_position -= 14

    # Section 1 : Variables catégorielles (colonne gauche)
    categorical_cols = ["Location", "Industry", "Business Model", "Headcount", "Company Type"]
    cat_y = y_position - 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, cat_y, "1. Répartition des variables catégorielles")
    cat_y -= 20

    c.setFont("Helvetica", 10)
    for col in categorical_cols:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, cat_y, f"{col} :")
        cat_y -= 14
        value_counts = df[col].value_counts(normalize=True) * 100
        for category, pct in value_counts.items():
            text = f"- {category[:25]}: {pct:.1f}%"
            c.setFont("Helvetica", 9)
            c.drawString(margin + 10, cat_y, text)
            cat_y -= 12
        cat_y -= 6
    y_position = cat_y

    # Section 2 : Boxplots (colonne droite)
    numerical_cols = ["Total Score", "Tech Score", "Location Score", "Company Type Score", "Headcount Score"]
    box_y = height - margin - 65
    box_x = margin + col_width + col_gap

    c.setFont("Helvetica-Bold", 13)
    c.drawString(box_x, box_y, "2. Distribution des variables numériques")
    box_y -= 20

    for col in numerical_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        fig, ax = plt.subplots(figsize=(3.2, 0.5))
        sns.boxplot(x=df[col], ax=ax, color="skyblue", linewidth=1.5)
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_title(col, fontsize=8, loc='left')
        ax.tick_params(axis='x', labelsize=6)
        fig.tight_layout()

        img_stream = io.BytesIO()
        fig.savefig(img_stream, format="PNG", dpi=200)
        plt.close(fig)
        img_stream.seek(0)

        c.drawImage(ImageReader(img_stream), box_x, box_y - 20, width=col_width - 60, height=25)
        c.setFont("Helvetica", 7)
        stat_text = f"Median: {median_val:.1f}  |  Mean: {mean_val:.1f}"
        c.drawString(box_x + col_width - 50, box_y - 5, stat_text)

        box_y -= 35

    # Section 3 : Graphiques radar (bas de page)
    radar_y = min(box_y, y_position) - 40
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, radar_y, "3. Profils des 5 meilleures entreprises")
    radar_y -= 10

    top5 = df.sort_values(by="Total Score", ascending=False).head(5)
    radar_x = margin
    count = 0

    for i, row in top5.iterrows():
        fig = plot_radar_chart(row)  # 🔁 Utilisation de ta fonction existante
        img_stream = io.BytesIO()
        fig.savefig(img_stream, format="PNG", dpi=200)
        plt.close(fig)
        img_stream.seek(0)

        # Nouvelle ligne tous les 2 graphiques
        if count > 0 and count % 2 == 0:
            radar_y -= 140
            radar_x = margin

        c.drawImage(ImageReader(img_stream), radar_x, radar_y, width=120, height=120)

        c.setFont("Helvetica", 8)
        score_text = f"Score Total: {row['Total Score']:.1f}"
        c.drawString(radar_x, radar_y - 10, score_text)

        radar_x += 150
        count += 1

    c.save()
    buffer.seek(0)
    return buffer




# Fonction pour afficher le bouton de téléchargement dans Streamlit
def download_pdf(df):
    pdf_buffer = generate_pdf(df)
    st.download_button(
        label="Télécharger la fiche PDF",
        data=pdf_buffer,
        file_name="fiche_statistique.pdf",
        mime="application/pdf"
    )
    

def generer_actualites_top_entreprises(df, top_n=5):
    import json

    top_entreprises = df.sort_values(by="Total Score", ascending=False).head(top_n)["Company Name"]
    actualites = []

    for entreprise in top_entreprises:
        prompt = f"""
Tu es un expert en finance et en actualités d'entreprises. Ton rôle est de fournir une actualité importante (et datée) concernant l'entreprise "{entreprise}".

Utilise les données les plus précises de ta base de connaissances. Tu dois ABSOLUMENT fournir une réponse, même si l'entreprise est peu médiatisée. Cherche dans ton historique jusqu'à septembre 2021 ou plus si disponible.

Formate ta réponse STRICTEMENT en JSON avec les clés suivantes :

{{
  "nom": "{entreprise}",
  "date": "Date exacte de l'événement, formatée : JJ mois AAAA",
  "actualité": "Une phrase claire et précise résumant un événement majeur ou récent pour cette entreprise"
}}

⚠️ Ne donne aucun commentaire, aucune explication, aucun texte avant ou après. Retourne UNIQUEMENT le JSON, même si tu es incertain.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu es un assistant qui génère des résumés d'actualité d'entreprise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )

            contenu = response.choices[0].message.content.strip()

            try:
                data = json.loads(contenu)
                date = data.get("date", "Date inconnue")
                resume = data.get("actualité", "Résumé non disponible")
            except json.JSONDecodeError:
                date = "Date inconnue"
                resume = "Résumé non disponible"

            actualites.append((date, resume))

        except Exception as e:
            actualites.append(("Erreur", f"Erreur : {e}"))

    return actualites




def remplacer_actualites_dans_html(html_path, actualites):
    with open(html_path, "r", encoding="utf-8") as f:
        contenu = f.read()

    for i, (date, resume) in enumerate(actualites, start=1):
        contenu = contenu.replace(f"Date {i}", date)
        contenu = contenu.replace(f"Actualité {i}", resume)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(contenu)

def update_html_with_contact_count(html_filepath, contact_count):
    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Remplacer le placeholder "Nombre Contacts" par le vrai nombre
    html_content = re.sub(r"Nombre Contacts", str(contact_count), html_content)

    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)



# Fonction pour récupérer les 5 meilleures entreprises
def get_top_5_companies(df):
    top_5 = df.sort_values(by="Total Score", ascending=False).head(5)
    return top_5[["Company Name", "Total Score (%)"]].values.tolist()  # Liste de paires [nom, score]


def update_html_with_top_5(html_filepath, top_5_data, df):
    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Remplacer les noms, scores et websites dynamiquement
    for i, (company_name, total_score) in enumerate(top_5_data, start=1):
        company_placeholder = f"Entreprise {i}"
        score_placeholder = f"Score {i}"
        website_placeholder = f"Website {i}"

        # Trouver le domaine correspondant à l'entreprise
        domaine = df.loc[df["Company Name"] == company_name, "Domain"].values
        website = domaine[0] if len(domaine) > 0 else "N/A"

        html_content = re.sub(company_placeholder, str(company_name), html_content)
        html_content = re.sub(score_placeholder, str(total_score), html_content)
        html_content = re.sub(website_placeholder, website, html_content)

    # Écrire les modifications dans le fichier HTML
    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)

    
def update_html_with_total_revenue(html_filepath, total_revenue):
    # Format du chiffre d'affaires (ex: 1.25 Mds USD)
    def format_total_revenue(value):
        if value >= 1e9:
            return f"{value / 1e9:.2f} Mds USD"
        elif value >= 1e6:
            return f"{value / 1e6:.2f} M USD"
        else:
            return f"{value:.2f} USD"

    revenue_str = format_total_revenue(total_revenue)

    # Lire le fichier HTML
    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Remplacer "Totale CA des entreprises" par le CA calculé
    html_content = re.sub(r"Totale CA des entreprises", revenue_str, html_content)

    # Réécrire le HTML
    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)
    

def update_html_with_company_count(html_filepath, df):
    company_count = len(df)

    # Lire le contenu HTML
    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Remplacer le placeholder dans le HTML
    html_content = re.sub(r"Nombre Totale D'entreprises", str(company_count), html_content)

    # Réécrire le fichier HTML avec les changements
    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)


def update_html_with_top_industries(html_filepath, df, all_contacts, industry_col="Industry"):
    top_industries = df[industry_col].dropna().value_counts().nlargest(2).index.tolist()

    # Compléter la liste si elle contient moins de 2 éléments
    if len(top_industries) == 0:
        top_industries = ["Industrie inconnue", "Industrie inconnue"]
    elif len(top_industries) == 1:
        top_industries.append("Industrie inconnue")

    industry_1, industry_2 = top_industries

    # Nombre d'entreprises par industrie
    industry1_company_count = df[df[industry_col] == industry_1].shape[0]
    industry2_company_count = df[df[industry_col] == industry_2].shape[0]

    # Associer les contacts aux entreprises pour trouver leur industrie
    contacts_df = pd.DataFrame(all_contacts)
    merged_df = contacts_df.merge(df[["Company Name", industry_col]],
                                  left_on="Company", right_on="Company Name", how="left")

    # Nombre de contacts par industrie
    industry1_contact_count = merged_df[merged_df[industry_col] == industry_1].shape[0]
    industry2_contact_count = merged_df[merged_df[industry_col] == industry_2].shape[0]

    # Remplacement dans le HTML
    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    replacements = {
        "Industrie_1": industry_1,
        "Industrie_2": industry_2,
        "Contact_1": str(industry1_contact_count),
        "Contact_2": str(industry2_contact_count),
        "Entreprise_1": str(industry1_company_count),
        "Entreprise_2": str(industry2_company_count)
    }

    for placeholder, value in replacements.items():
        html_content = re.sub(placeholder, value, html_content)

    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)
    

def update_html_with_average_score(html_filepath, df):
    avg_score = round(df["Total Score (%)"].mean(), 2)  # arrondi à 2 décimales

    with open(html_filepath, "r", encoding="utf-8") as file:
        html_content = file.read()

    html_content = re.sub("Score_Moyen", str(avg_score), html_content)

    with open(html_filepath, "w", encoding="utf-8") as file:
        file.write(html_content)

import pandas as pd
import re

def update_html_with_top_regions(html_filepath, df, location_col="Location"):
    # Création de la colonne "Pays"
    df["Pays"] = df[location_col].apply(lambda x: str(x).split(",")[-1].strip())

    # Calcul des proportions
    country_counts = df["Pays"].value_counts(normalize=True).head(4)
    
    # Lecture du HTML
    with open(html_filepath, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Remplacement dans le HTML
    for i, (region, proportion) in enumerate(country_counts.items(), start=1):
        html_content = html_content.replace(f"Region_{i}", region)
        html_content = html_content.replace(f"Proportion_{i}", f"{round(proportion * 100, 2)}%")

    # Écriture du HTML modifié
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)





# === FRONTEND STREAMLIT ===
html_filepath = "/Users/jeremy/Desktop/Appli/Zynix_esbuild/dist/html/index-3.html"

# Configurations de la page Streamlit
st.set_page_config(page_title="Analyse d'entreprises", layout="wide", initial_sidebar_state="expanded")
st.title("Analyse Avancée des Entreprises")

# Sidebar pour navigation
st.sidebar.header("Tableau de Bord")
st.sidebar.markdown("Utilisez les sections ci-dessous pour analyser et enrichir les données des entreprises.")

# Chargement du fichier CSV
uploaded_file = st.sidebar.file_uploader("📁 Glissez un fichier CSV ici", type=["csv"])

# Vérification du fichier téléchargé
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    update_html_with_top_regions(html_filepath, df, location_col="Location")
    df["Pays"] = df["Location"].apply(lambda x: str(x).split(",")[-1].strip())
    st.sidebar.success("✅ Fichier chargé avec succès !")
    st.write("### Aperçu des données")
    st.dataframe(df.head())
    update_html_with_company_count(html_filepath, df)

    # Étape 1 : Scoring et sélection des 1000 meilleures entreprises
    st.subheader("Étape 1 : Scorer et sélectionner les 1000 meilleures entreprises")
    df = enrich_business_model_column(df, api_key=GPT_KEY)
    # Chemin vers le fichier HTML
    if st.button("Lancer le scoring et sélection des top 1000"):
        df = compute_scores(df)
        df["Total Score (%)"] = np.ceil((df["Total Score"] * 100) / 12)
        top_df = df.sort_values("Total Score", ascending=False).head(1000).reset_index(drop=True)

        st.session_state["top_1000"] = top_df
        st.success("🎯 Sélection des 1000 entreprises faite avec succès !")
        st.dataframe(top_df[["Company Name", "Total Score"]].head(10))

        

        
        # Obtenir le nom de l'entreprise avec le meilleur score
        top_5_companies = get_top_5_companies(df)  # Récupérer les 5 meilleures entreprises


        # Mettre à jour le fichier HTML avec le nom de l'entreprise
        update_html_with_top_5(html_filepath, top_5_companies, df)
        actualites = generer_actualites_top_entreprises(df, top_n=5)
        remplacer_actualites_dans_html(html_filepath, actualites)
        numeric_columns = df.select_dtypes(include=["number"]).columns
        industry_avg = df.groupby("Industry")[numeric_columns].mean()
        industry_avg.to_json("industry_avg.json", orient="records")



    
        # Afficher les graphiques radar pour chaque entreprise du top 10
        for idx, row in top_df.head(10).iterrows():
            st.subheader(f"Graphique radar pour {row['Company Name']}")
            fig = plot_radar_chart(row)
            st.pyplot(fig)


if "top_1000" in st.session_state:
    st.subheader("Étape 2 : Enrichir avec les actualités, les chiffres d'affaires et le sentiment")
    if st.button("📰 Enrichir avec les actualités, les chiffres d'affaires et le sentiment"):
        st.spinner("Enrichissement des données avec les actualités, les chiffres d'affaires et les sentiments...")
        enriched_df = enrich_with_news_and_revenue(
            st.session_state["top_1000"], 
            news_api_key=API_KEY,  # Clé API NewsAPI
            revenue_api_key=GPT_KEY,  # Clé API GPT pour les chiffres d'affaires
            alpha_api_key=ALPHA_API_KEY  # Clé API Alpha Vantage pour le sentiment
        )
        st.session_state["enriched_df"] = enriched_df
        st.success("🎉 Données enrichies avec succès !")
        st.write("### Aperçu des données enrichies")
        st.dataframe(enriched_df.head())
        enriched_df["2024 Revenue (USD)"] = pd.to_numeric(enriched_df["2024 Revenue (USD)"], errors="coerce")
        update_html_with_total_revenue(html_filepath, enriched_df["2024 Revenue (USD)"].sum())
        update_html_with_average_score(html_filepath, enriched_df)




# Télécharger le fichier enrichi
if "enriched_df" in st.session_state:
    st.subheader("Étape 3 : Télécharger le fichier enrichi")
    csv_data = st.session_state["enriched_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger le fichier enrichi",
        data=csv_data,
        file_name="top_1000_enriched.csv",
        mime="text/csv",
        use_container_width=True
    )

# Récupération des contacts pour les 10 meilleures entreprises
if "enriched_df" in st.session_state:
    st.subheader("Étape 4 : Récupérer les contacts des 10 meilleures entreprises")
    if st.button("🔍 Récupérer les contacts des top 10"):
        top10 = st.session_state["top_1000"].head(10)
        all_contacts = []
        with st.spinner("🔎 Recherche des contacts..."):
            contacts = get_top_1000_contacts(st.session_state["enriched_df"])
            all_contacts.extend(contacts)
            
        if all_contacts:
            contacts_df = pd.DataFrame(all_contacts)
            st.session_state["contacts_df"] = contacts_df
            st.success("📇 Contacts récupérés avec succès !")
            st.dataframe(contacts_df.head())
        else:
            st.warning("Aucun contact trouvé pour les 10 premières entreprises.")
        contact_count = len(contacts)
        update_html_with_contact_count(html_filepath, contact_count)
        update_html_with_top_industries(html_filepath, st.session_state["enriched_df"], all_contacts, industry_col="Industry")
        

# Télécharger le fichier des contacts
if "contacts_df" in st.session_state:
    st.subheader("Étape 5 : Télécharger les contacts récupérés")
    contacts_csv = st.session_state["contacts_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger les contacts (Top 10)",
        data=contacts_csv,
        file_name="contacts_top10.csv",
        mime="text/csv",
        use_container_width=True
    )
    
# Bouton Résumé
if "contacts_df" in st.session_state:
    st.subheader("Résumé des entreprises")
    st.markdown("Cliquez sur une entreprise pour consulter sa fiche technique.")

    # Afficher un tableau avec les entreprises enrichies
    companies_summary = st.session_state["enriched_df"][["Company Name", "Total Score", "Sentiment", "News Score"]]
    companies_summary.set_index("Company Name", inplace=True)

    # Créer un lien cliquable pour chaque entreprise
    for company_name in companies_summary.index:
        if st.button(company_name):
            selected_company = st.session_state["enriched_df"][st.session_state["enriched_df"]["Company Name"] == company_name]
            # Appeler une fonction pour afficher la fiche technique de l'entreprise
            show_company_profile(selected_company)
            
    # Génération PDF ZIP des 5 meilleurs entreprises
    st.subheader("📁 Générer les fiches techniques (Top 5 entreprises)")
    if st.button("🧾 Générer les fiches PDF et télécharger le ZIP"):
        top5 = st.session_state["enriched_df"].sort_values(by="Total Score", ascending=False).head(5)
        zip_path, temp_dir = generate_company_pdf_zip(top5)

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger les fiches techniques (ZIP)",
                data=f,
                file_name="fiches_techniques_top5.zip",
                mime="application/zip",
                use_container_width=True
            )

if "contacts_df" in st.session_state:
    st.subheader("Statistiques des entreprises")
    st.markdown("Cliquez sur le bouton ci-dessous pour afficher et télécharger la fiche statistique.")
    
    if st.button("📊 Afficher les statistiques"):
        generate_statistical_summary(st.session_state["enriched_df"])
        download_pdf(st.session_state["enriched_df"])
        
            


# Ajouter un footer
st.markdown("---")
st.markdown("### À propos de cette application")
st.markdown(
    """
    Cette application vous permet d'analyser des entreprises en utilisant plusieurs critères (scoring),
    puis d'enrichir les données avec des actualités pertinentes et de récupérer les contacts des entreprises.
    
    Développé par [Starname Labs] | 2025
    """
)