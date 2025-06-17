import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import pipeline
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import timezone
import pytz  # Pour la conversion de fuseaux horaires
import time

# Titre de l'application
st.title("📊 Sentimètre")
st.markdown("Analysez les émotions autour d’un mot-clé sur X (Twitter).")

# Zone de saisie utilisateur
keyword = st.text_input("Entrez un mot-clé ou hashtag", "#IA")
max_tweets = st.slider("Nombre de tweets à analyser", 10, 500, 100)

# Pipeline HuggingFace pour analyse fine des émotions
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

emotion_classifier = load_emotion_model()

# Fonction pour récupérer les tweets
def collecter_tweets(keyword, max_tweets):
    tweets_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if i >= max_tweets:
            break
        tweets_list.append({
            "text": tweet.content,
            "datetime": tweet.date
        })
        time.sleep(0.01)
    return pd.DataFrame(tweets_list)

# Fonction d'analyse des émotions
def analyser_emotions(df):
    df['emotion'] = df['text'].apply(lambda x: emotion_classifier(x)[0]['label'])
    return df

# Convertir en heure locale (exemple : Paris - UTC+1)
def convertir_en_heure_locale(df):
    tz = pytz.timezone('Europe/Paris')
    df['heure_locale'] = df['datetime'].dt.tz_localize(timezone.utc).dt.tz_convert(tz)
    df['heure'] = df['heure_locale'].dt.floor('H')  # Arrondir à l’heure
    return df

# Créer une courbe avec moyenne mobile
def creer_graphique_evolution(df):
    # Compte les émotions par heure
    df_grouped = df.groupby(['heure', 'emotion']).size().reset_index(name='count')

    # Trier par date
    df_grouped = df_grouped.sort_values('heure')

    # Appliquer une moyenne mobile glissante (fenêtre de 3 heures)
    df_smooth = df_grouped.groupby(['emotion']).apply(
        lambda x: x.set_index('heure').rolling('24h', min_periods=1)['count'].mean()
    ).reset_index()

    df_smooth.rename(columns={None: 'moving_avg'}, inplace=True)

    fig = px.line(
        df_smooth,
        x='heure',
        y='moving_avg',
        color='emotion',
        title="Évolution des émotions au fil du temps (avec moyenne mobile)",
        labels={'heure': 'Date / Heure', 'moving_avg': 'Tweets moyens par heure'}
    )
    fig.update_layout(hovermode="x unified")
    return fig

# Lancer l'analyse au clic
if st.button("Lancer l'analyse"):
    with st.spinner("Collecte et analyse en cours..."):
        df = collecter_tweets(keyword, max_tweets)
        df = analyser_emotions(df)
        df = convertir_en_heure_locale(df)

        # Afficher les résultats
        st.subheader("🧠 Répartition des émotions")
        fig_hist = px.histogram(df, x='emotion', color='emotion', title="Distribution des émotions ressenties")
        st.plotly_chart(fig_hist)

        # Nuage de mots
        all_text = " ".join(df['text'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # Courbe temporelle
        st.subheader("🕒 Évolution des émotions au fil du temps")
        fig_line = creer_graphique_evolution(df)
        st.plotly_chart(fig_line, use_container_width=True)

        # Tableau des tweets
        st.subheader("📄 Tweets analysés")
        st.dataframe(df[['heure_locale', 'emotion', 'text']].head(20))

        # Export CSV
        st.download_button(
            label="📥 Télécharger les résultats",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='sentimetre_resultats.csv',
            mime='text/csv'
        )