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
news_api_key = 'a5a5a71c72b74b2a9e09e9ed9af542ca'
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
    df["Total Score"] = [score[6] for score in scores]
    df["Total Score %"] = np.ceil((df["Total Score"] * 100) / 12)
    
    return df




# Fonction pour récupérer les news
def get_news(company_name):
    today = datetime.today().strftime('%Y-%m-%d')
    seven_days_ago = (datetime.today() - timedelta(days=31)).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q={company_name}&from={seven_days_ago}&to={today}&sortBy=publishedAt&apiKey={news_api_key}'
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




# Fonction pour enrichir avec les actualités, les chiffres d'affaires 2024 et le sentiment
def enrich_with_news_and_revenue(df, news_api_key, revenue_api_key):
    # Enrichir avec les actualités
    features = ['Nouveau Directeur', 'Levee de Fonds', 'Acquisition/Fusion', 'Expansion Géographique', 'Lancement Produit']
    for f in features:
        df[f] = 'NaN'
    df['News Score'] = 0

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

        time.sleep(1)  # Respect du rate limit de l'API NewsAPI

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
        "Chaque ligne doit être un nombre entier, sans décimales. Ne me dis rien d'autre aucune phrase du style Il est impossible d'obtenir ces données. C'est juste une estimation du CA en 2024 et même c'est pas grave si tu te trompes. Donnes juste une valeurs pour chaque entreprises. Merci beauoup"
    )

    try:
        # Appel à l'API OpenAI
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {revenue_api_key}"
        }
        data = {
            "model": "gpt-4",  # Utilisez "gpt-4" ou "gpt-4-32k" si nécessaire
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            # Traiter la réponse de l'API
            content = response.json()['choices'][0]['message']['content']
            print(f"Réponse de l'API : {content}")
            revenues = content.strip().split('\n')

            # Vérifier que la longueur correspond
            if len(revenues) == len(companies):
                # Nettoyer les données et les convertir en nombres entiers
                revenues = [int(revenue.strip()) if revenue.strip().isdigit() else None for revenue in revenues]
                df['2024 Revenue (USD)'] = pd.Series(revenues)
                print("✅ Colonne '2024 Revenue (USD)' ajoutée avec succès.")
            else:
                raise Exception("Mauvaise longueur de réponse de l’API.")
        else:
            raise Exception(f"Erreur API OpenAI : {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ Erreur enrichissement chiffre d'affaires : {e}")
        # Ajouter une colonne par défaut si l'API échoue
        df['2024 Revenue (USD)'] = None

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
    + "\n\nPour chaque entreprise, retourne uniquement une des valeurs suivantes : B2B, B2C ou B2B et B2C. "
    "Chaque ligne doit contenir uniquement une de ces valeurs, sans texte supplémentaire. "
    f"Assure-toi que la réponse contient exactement {len(companies)} lignes, "
    "une pour chaque entreprise, dans le même ordre que la liste fournie."
    )

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4-turbo",
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


def main(input_csv_path, output_csv_path, output_json_path, contacts_csv_path, contacts_json_path):
    # Charger le fichier CSV
    print(f"📂 Chargement du fichier CSV : {input_csv_path}")
    df = pd.read_csv(input_csv_path, on_bad_lines='skip')
    print(f"✅ Fichier chargé avec succès. Colonnes disponibles : {df.columns.tolist()}")

    # Étape 1 : Enrichir avec les modèles d'affaires si non présent
    print("🔄 Étape 1 : Enrichissement avec les modèles d'affaires...")
    df = enrich_business_model_column(df, GPT_KEY)
    print(f"✅ Colonnes après enrichissement des modèles d'affaires : {df.columns.tolist()}")
    print(df.head())

    # Étape 2 : Calculer les scores
    print("🔄 Étape 2 : Calcul des scores...")
    df = compute_scores(df)
    print(f"✅ Colonnes après calcul des scores : {df.columns.tolist()}")
    print(df.head())

    # Étape 3 : Enrichir avec news, revenus, et sentiment
    print("🔄 Étape 3 : Enrichissement avec les actualités, revenus et sentiment...")
    df = enrich_with_news_and_revenue(df, news_api_key, GPT_KEY)
    print(f"✅ Colonnes après enrichissement avec les actualités : {df.columns.tolist()}")
    print(df.head())

    # Étape 4 : Générer les actualités "News" pour les top entreprises
    print("🔄 Étape 4 : Génération des actualités pour les top entreprises...")
    actualites = generer_actualites_top_entreprises(df)
    top_names = df.sort_values(by="Total Score", ascending=False).head(len(actualites))["Company Name"].tolist()
    news_dict = dict(zip(top_names, actualites))
    print(f"✅ Actualités générées pour les entreprises : {news_dict}")

    # Ajouter les actualités dans une colonne "News"
    df["News"] = df["Company Name"].apply(lambda name: f"{news_dict.get(name, ('Date inconnue', 'Pas d actualité'))[0]} - {news_dict.get(name, ('', 'Pas d actualité'))[1]}")
    df["Date"] = df["Company Name"].apply(lambda name: news_dict.get(name, ("Date inconnue", "Résumé non disponible"))[0])

    # Sauvegarde du fichier enrichi au format CSV
    print(f"💾 Sauvegarde du fichier enrichi au format CSV à : {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Fichier enrichi sauvegardé avec succès au format CSV.")

    # Sauvegarde du fichier enrichi au format JSON
    print(f"💾 Sauvegarde du fichier enrichi au format JSON à : {output_json_path}")
    df.to_json(output_json_path, orient="records", indent=4, force_ascii=False)
    print(f"✅ Fichier enrichi sauvegardé avec succès au format JSON.")

    # Étape 5 : Récupérer les contacts des 1000 meilleures entreprises
    print("🔄 Étape 5 : Récupération des contacts des 1000 meilleures entreprises...")
    contacts = get_top_1000_contacts(df, HUNTER_API_KEY)

    # Sauvegarder les contacts dans un fichier CSV
    print(f"💾 Sauvegarde des contacts au format CSV à : {contacts_csv_path}")
    contacts_df = pd.DataFrame(contacts)
    contacts_df.to_csv(contacts_csv_path, index=False)
    print(f"✅ Fichier des contacts sauvegardé avec succès au format CSV.")

    # Sauvegarder les contacts dans un fichier JSON
    print(f"💾 Sauvegarde des contacts au format JSON à : {contacts_json_path}")
    contacts_df.to_json(contacts_json_path, orient="records", indent=4, force_ascii=False)
    print(f"✅ Fichier des contacts sauvegardé avec succès au format JSON.")


if __name__ == "__main__":
    input_path = "test.csv"  # Ton fichier d'entrée
    output_csv_path = "companies_enriched.csv"  # Fichier enrichi au format CSV
    output_json_path = "companies_enriched.json"  # Fichier enrichi au format JSON
    contacts_csv_path = "contacts.csv"  # Fichier des contacts au format CSV
    contacts_json_path = "contacts.json"  # Fichier des contacts au format JSON
    main(input_path, output_csv_path, output_json_path, contacts_csv_path, contacts_json_path)






    













