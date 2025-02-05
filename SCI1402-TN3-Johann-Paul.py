#!/usr/bin/env python
# coding: utf-8

# # Tableau de bord interactif avec StreamLite

# In[7]:


import os

# Définir le répertoire de travail
directory = "C:/Teluq/Automne 2024/SCI 1402 - Projet en science des données/SCI1402-TN3-Johann-Paul"
os.chdir(directory)  


# In[8]:


import streamlit as st
import pandas as pd
import folium
import plotly.express as px
import xgboost as xgb
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
import sys

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("crimes_par_quartier.csv")
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df["ANNEE"] = df["DATE"].dt.year
    df["MOIS"] = df["DATE"].dt.month
    df["JOUR"] = df["DATE"].dt.day
    return df

df = load_data()

# Interface utilisateur
st.title("📍 Analyse et Prédiction des Crimes à Montréal")
st.sidebar.header("🔎 Filtres")

# Sélection du type de crime
crime_types = df["CATEGORIE"].unique()
selected_crime = st.sidebar.selectbox("Sélectionner un type de crime", crime_types)

# Sélection du quartier
quartiers = df["nom_qr"].dropna().unique()
selected_quartier = st.sidebar.selectbox("Filtrer par quartier", ["Tous"] + list(quartiers))

# Partie Prédiction
st.sidebar.subheader("🚀 Prédictions de Crimes")

# Sélection du modèle de Machine Learning
model_choice = st.sidebar.selectbox(
    "🧠 Choisir un modèle d'IA",
    ["XGBoost", "Régression Logistique", "Forêt Aléatoire", "SVM"]
)

# Définir si une prédiction est demandée
prediction_demandee = st.sidebar.button("📊 Prédire le type de crime et son évolution")

if prediction_demandee:
    try:
        # Vérification des données de quartier
        if selected_quartier == "Tous":
            st.sidebar.error("⚠️ Veuillez sélectionner un quartier spécifique pour la prédiction.")
            st.stop()

        if selected_quartier not in quartiers:
            st.sidebar.error("❌ Quartier non trouvé dans les données. Veuillez choisir parmi la liste.")
            st.stop()

        # 🏙️ Encodage des catégories
        le_crime = LabelEncoder()
        le_quart = LabelEncoder()

        df["CATEGORIE"] = le_crime.fit_transform(df["CATEGORIE"])
        df["nom_qr"] = le_quart.fit_transform(df["nom_qr"].astype(str))

        # Préparation des données
        features = ["ANNEE", "MOIS", "JOUR", "nom_qr"]
        X = df[features]
        y = df["CATEGORIE"]
        
        # Initialiser le DataFrame vide pour stocker les prédictions
        predictions_df = pd.DataFrame()

        # Initialisation du modèle choisi
        if model_choice == "XGBoost":
            model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(y.unique()), eval_metric="mlogloss")
        elif model_choice == "Régression Logistique":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Forêt Aléatoire":
            model = RandomForestClassifier(n_estimators=100)
        elif model_choice == "SVM":
            model = SVC(probability=True)

        # Entraîner le modèle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Prédiction sur 12 mois
        future_dates = pd.date_range(start=pd.to_datetime("today"), periods=12, freq='MS')
        predictions = []

        for date in future_dates:
            future_data = [[date.year, date.month, 15, le_quart.transform([selected_quartier])[0]]]
            pred = model.predict(future_data)[0]
            predictions.append({
                "Date": date.strftime("%Y-%m"),
                "Crime Prédit": le_crime.inverse_transform([pred])[0],
                "Probabilité": model.predict_proba(future_data)[0].max() if hasattr(model, "predict_proba") else "N/A"
            })
            
        # Stocker les résultats pour l'exportation
        predictions_df = pd.DataFrame(predictions)

        # Affichage des résultats
        predictions_df = pd.DataFrame(predictions)
        st.subheader(f"🔮 Prévisions pour {selected_quartier}")
                        
        # Détails des prédictions
        st.dataframe(predictions_df.style.format({"Probabilité": "{:.2%}"}))
        
        # Graphique temporel
        fig = px.line(predictions_df, x='Date', y='Probabilité', title='Évolution des probabilités de crime')
        st.plotly_chart(fig)

        # Affichage de la précision du modèle
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.sidebar.info(f"📊 **Modèle utilisé : {model_choice}**")
        st.sidebar.success(f"✅ **Précision du modèle : {accuracy:.2f}**")

    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")
        st.info("Veuillez sélectionner un quartier avec suffisamment de données historiques")
        
     #  Export des données filtrées 
    st.sidebar.subheader("📥 Exporter les données filtrées")
    st.sidebar.download_button(
        label="📂 Télécharger CSV",
        data=df.to_csv(index=False),
        file_name="crimes_filtrés.csv",
        mime="text/csv"
    )

    if not predictions_df.empty:
        st.sidebar.subheader("📥 Exporter les données prédites")
        st.sidebar.download_button(
            label="📂 Télécharger les prédictions",
            data=predictions_df.to_csv(index=False),
            file_name="predictions_crimes.csv",
            mime="text/csv"
    )
    
#  Si une prédiction est demandée, ne pas afficher la carte et l'évolution des crimes
if not prediction_demandee:
    if selected_quartier == "Tous":
        df_filtered = df[df["CATEGORIE"] == selected_crime]
    else:
        df_filtered = df[(df["CATEGORIE"] == selected_crime) & (df["nom_qr"] == selected_quartier)]

    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée disponible pour cette combinaison type de crime/quartier")
    else:
        st.subheader(f"🗺️ Carte des crimes {selected_crime} pour {selected_quartier}")
        m = folium.Map(location=[45.6, -73.7], zoom_start=11)

        for _, row in df_filtered.iterrows():
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.6,
                popup=f"{row['CATEGORIE']} - {row['DATE'].date()}",
            ).add_to(m)

        HeatMap([[row["LATITUDE"], row["LONGITUDE"]] for _, row in df_filtered.iterrows()]).add_to(m)
        folium_static(m)
        
        # Graphique d'évolution temporelle
        st.subheader("📈 Évolution des crimes au fil du temps")
        df_grouped = df_filtered.resample("M", on="DATE").size().reset_index(name="Nombre de crimes")
        fig = px.line(df_grouped, x="DATE", y="Nombre de crimes", title=f"Évolution des {selected_crime}")
        st.plotly_chart(fig)
        
        #  Indicateurs statistiques
        st.subheader("📊 Indicateurs statistiques")
        st.write(f"🔹 Nombre total de crimes : {len(df_filtered)}")
        st.write(f"🔹 Nombre moyen de crimes par mois : {df_grouped['Nombre de crimes'].mean():.2f}")
        
#  Export des données filtrées 
    st.sidebar.download_button(
        label="📂 Télécharger les données filtrées",
        data=df_filtered.to_csv(index=False),
        file_name="crimes_filtrés.csv",
        mime="text/csv"
    )
    
#  Bouton de fermeture
if st.sidebar.button("❌ Quitter l'application"):
    st.sidebar.warning("⚠️ L'application va se fermer.")
    time.sleep(3)
    st.sidebar.warning("Vous pouvez maintenant fermer cet onglet en toute sécurité.")
    st.stop()
    sys.exit()


# In[ ]:





# In[ ]:




