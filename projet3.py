import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# Fonction pour lire les données
def load_data():
    # Charger les données des fichiers CSV
    weather_df = pd.read_csv('weather_meteo.csv')
    circuits_df = pd.read_csv('cleaned_circuits.csv')
    
    # Afficher les types de données pour vérification
    print("Types de colonnes dans weather_df :", weather_df.dtypes)
    print("Types de colonnes dans circuits_df :", circuits_df.dtypes)
    
    # Nettoyer et convertir les données de latitude et longitude
    if 'fact_latitude' in weather_df.columns and 'fact_longitude' in weather_df.columns:
        # Vérifier le type des colonnes avant conversion
        if weather_df['fact_latitude'].dtype == object:
            weather_df['fact_latitude'] = pd.to_numeric(weather_df['fact_latitude'].str.replace(',', '.'), errors='coerce')
        if weather_df['fact_longitude'].dtype == object:
            weather_df['fact_longitude'] = pd.to_numeric(weather_df['fact_longitude'].str.replace(',', '.'), errors='coerce')
    else:
        raise KeyError("Les colonnes 'fact_latitude' et 'fact_longitude' sont absentes dans weather_meteo.csv.")
    
    if 'lat' in circuits_df.columns and 'lng' in circuits_df.columns:
        # Vérifier le type des colonnes avant conversion
        if circuits_df['lat'].dtype == object:
            circuits_df['lat'] = pd.to_numeric(circuits_df['lat'].str.replace(',', '.'), errors='coerce')
        if circuits_df['lng'].dtype == object:
            circuits_df['lng'] = pd.to_numeric(circuits_df['lng'].str.replace(',', '.'), errors='coerce')
    else:
        raise KeyError("Les colonnes 'lat' et 'lng' sont absentes dans cleaned_circuits.csv.")
    
    return weather_df, circuits_df

# Fonction pour associer les conditions météo aux circuits
def match_weather_to_circuits(weather_df, circuits_df):
    # Convertir les coordonnées en format numpy
    weather_coords = weather_df[['fact_latitude', 'fact_longitude']].values
    circuit_coords = circuits_df[['lat', 'lng']].values
    
    # Utiliser NearestNeighbors pour trouver les circuits les plus proches des conditions météo
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(circuit_coords)
    distances, indices = nn.kneighbors(weather_coords)
    
    # Ajouter les informations de météo aux circuits
    weather_df['nearest_circuit'] = indices.flatten()
    weather_df['circuit_lat'] = circuits_df.loc[weather_df['nearest_circuit'], 'lat'].values
    weather_df['circuit_lng'] = circuits_df.loc[weather_df['nearest_circuit'], 'lng'].values
    
    return weather_df

# Fonction pour générer des données fictives avec randomisation
def generate_random_data(weather_df, seed=None):
    drivers = [
        "Alexander Albon", "Carlos Sainz", "Charles Leclerc", "Daniel Ricciardo", "Esteban Ocon", 
        "Fernando Alonso", "George Russell", "Guanyu Zhou", "Lance Stroll", "Lando Norris", 
        "Lewis Hamilton", "Logan Sargeant", "Max Verstappen", "Nico Hülkenberg", "Oscar Piastri", 
        "Pierre Gasly", "Sergio Pérez", "Valtteri Bottas", "Yuki Tsunoda"
    ]
    
    courses = [
        "Bahreïn (Sakhir)", "Australie (Melbourne)", "Chine (Shanghai)", "Japon (Suzuka)",
        "Arabie saoudite (Djeddah)", "Miami (Floride)", "Emilie Romagne (Imola – Italie)",
        "Monaco (Monte-Carlo)", "Espagne (Barcelone)", "Canada (Montréal)", "Autriche (Spielberg)",
        "Royaume-Uni (Silverstone)", "Hongrie (Budapest)", "Grand Prix de Belgique (Spa-Francorchamps)",
        "Pays-Bas (Zandvoort)", "Italie (Monza)", "Azerbaïdjan (Bakou)", "Singapour (Marina Bay)",
        "USA (Austin)", "Mexique (Mexico City)", "Brésil (São Paulo)", "Las Vegas (Nevada)",
        "Qatar (Doha)", "Abu Dhabi (Yas Marina)"
    ]
    
    # Utiliser un seed si fourni pour des résultats reproductibles
    if seed is not None:
        np.random.seed(seed)
    
    course_data = {}
    
    for course in courses:
        if course == "Arabie saoudite (Djeddah)":
            # Cas spécial pour Arabie saoudite (Djeddah)
            speeds = np.random.uniform(190, 210, len(drivers))  # Modifier les vitesses pour cet événement
            times = np.random.uniform(85, 95, len(drivers))  # Modifier les temps pour cet événement
        else:
            if course in weather_df['nearest_circuit'].values:
                # Obtenir les conditions météo pour le circuit le plus proche
                weather = weather_df[weather_df['nearest_circuit'] == course].iloc[0]  # Prendre la première ligne
                temperature_effect = 1 - (weather['temperature'] - 20) / 100  # Ajustements basés sur la température
                temperature_effect = np.clip(temperature_effect, 0.5, 1.5)  # Limiter l'effet
                
                speeds = np.random.uniform(200, 220, len(drivers)) * temperature_effect
                times = np.random.uniform(80, 100, len(drivers)) / temperature_effect
                times.sort()  # Assure que les temps sont triés pour un affichage réaliste
            else:
                speeds = np.random.uniform(200, 220, len(drivers))
                times = np.random.uniform(80, 100, len(drivers))
                times.sort()
        
        course_data[course] = {
            "Driver": drivers,
            "Average Fastest Lap Speed": speeds,
            "average_time_minutes": times
        }
    
    return course_data

# Charger les données
weather_df, circuits_df = load_data()
weather_df = match_weather_to_circuits(weather_df, circuits_df)
all_data = generate_random_data(weather_df)

# Sélectionner une course parmi la liste des courses
courses = list(all_data.keys())

st.title('Simulation de Course F1')
selected_course = st.selectbox("Choisissez une course", courses)

# Filtrer les données pour la course sélectionnée
if selected_course in all_data:
    course_data = all_data[selected_course]
    finish_times_df = pd.DataFrame(course_data)
else:
    finish_times_df = pd.DataFrame()

# Si aucune donnée pour la course sélectionnée
if finish_times_df.empty:
    st.warning(f"Aucune donnée trouvée pour {selected_course}.")
else:
    # Simulation de la course
    fig = go.Figure()
    
    # Ajouter une trace pour chaque pilote
    for i, row in finish_times_df.iterrows():
        # Créer une courbe de progression en fonction du temps pour chaque pilote
        fig.add_trace(go.Scatter(
            x=np.linspace(0, row['average_time_minutes'], 100),  # 100 points entre 0 et le temps moyen
            y=np.linspace(row['Average Fastest Lap Speed'], row['Average Fastest Lap Speed'] + (row['average_time_minutes'] / 2), 100),  # Simule une variation de vitesse
            mode='lines+markers',
            name=row['Driver'],
            line=dict(width=2)
        ))

    # Configuration du graphique
    fig.update_layout(
        title=f"Simulation de la course {selected_course}",
        xaxis_title="Temps (minutes)",
        yaxis_title="Vitesse (km/h)",
        legend_title="Pilotes"
    )

    # Afficher le graphique
    st.plotly_chart(fig)
    
    # Afficher le classement final
    st.subheader(f"Classement Final pour {selected_course}")
    st.dataframe(finish_times_df[['Driver', 'average_time_minutes']].rename(columns={'Driver': 'Pilote', 'average_time_minutes': 'Temps (minutes)'}))
