import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Charger et afficher le logo F1 avec une taille réduite
logo_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/F1-LOGO.png'
logo = Image.open(logo_path)
st.image(logo, width=300, caption="Simulation Formule 1")

# Charger les fichiers CSV avec le chemin correct sur votre machine
temps_par_courses_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/temps_par_courses.csv'
drivers_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/drivers.csv'
qualifying_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/cleaned_qualifying.csv'
weather_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/weather_meteo.csv'
circuits_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/cleaned_circuits.csv'

# Charger les données
temps_par_courses_df = pd.read_csv(temps_par_courses_path)
drivers_df = pd.read_csv(drivers_path)
qualifying_df = pd.read_csv(qualifying_path)
weather_df = pd.read_csv(weather_path)
circuits_df = pd.read_csv(circuits_path)

# Fonction pour nettoyer les coordonnées (latitude/longitude)
def clean_coordinates(df, lat_col, lng_col):
    df[lat_col] = df[lat_col].astype(str).str.replace(',', '.').astype(float)
    df[lng_col] = df[lng_col].astype(str).str.replace(',', '.').astype(float)
    return df

# Appliquer la fonction de nettoyage sur les colonnes de latitude et longitude
weather_df = clean_coordinates(weather_df, 'fact_latitude', 'fact_longitude')
circuits_df = clean_coordinates(circuits_df, 'lat', 'lng')

# S'assurer que la colonne 'milliseconds' est bien numérique
temps_par_courses_df['milliseconds'] = pd.to_numeric(temps_par_courses_df['milliseconds'], errors='coerce')
temps_par_courses_df = temps_par_courses_df.dropna(subset=['milliseconds'])

# Mapping des IDs de pilotes à leurs noms
driver_mapping = {
    830: "Max Verstappen",
    815: "Sergio Pérez",
    844: "Charles Leclerc",
    832: "Carlos Sainz",
    1: "Lewis Hamilton",
    847: "George Russell",
    4: "Fernando Alonso",
    840: "Lance Stroll",
    846: "Lando Norris",
    857: "Oscar Piastri",
    842: "Pierre Gasly",
    839: "Esteban Ocon",
    76: "Kevin Magnussen",
    807: "Nico Hülkenberg",
    848: "Alex Albon",
    858: "Logan Sargeant",
    822: "Valtteri Bottas",
    855: "Guanyu Zhou",
    817: "Daniel Ricciardo",
    852: "Yuki Tsunoda"
}

# Notes générales des pilotes actuelles
general_ratings = {
    "Max Verstappen": 92, 
    "Sergio Pérez": 88, 
    "Charles Leclerc": 89, 
    "Carlos Sainz": 88, 
    "Lewis Hamilton": 87, 
    "George Russell": 86, 
    "Fernando Alonso": 85, 
    "Lando Norris": 90, 
    "Oscar Piastri": 82, 
    "Pierre Gasly": 84, 
    "Esteban Ocon": 83, 
    "Kevin Magnussen": 78, 
    "Nico Hülkenberg": 80, 
    "Alex Albon": 82, 
    "Logan Sargeant": 70, 
    "Valtteri Bottas": 80, 
    "Guanyu Zhou": 75, 
    "Daniel Ricciardo": 81, 
    "Yuki Tsunoda": 78,
    "Lance Stroll": 75
}


# Moyennes des positions de départ et d'arrivée (en fonction de tes données historiques)
average_starting_positions = {
    "Max Verstappen": 3.75, 
    "Sergio Pérez": 7.20, 
    "Charles Leclerc": 5.80, 
    "Carlos Sainz": 6.50, 
    "Lewis Hamilton": 4.00, 
    "George Russell": 8.50, 
    "Fernando Alonso": 7.80, 
    "Lando Norris": 6.00, 
    "Oscar Piastri": 8.00, 
    "Pierre Gasly": 9.50, 
    "Esteban Ocon": 10.00, 
    "Kevin Magnussen": 12.00, 
    "Nico Hülkenberg": 11.00, 
    "Alex Albon": 10.50, 
    "Logan Sargeant": 18.00, 
    "Valtteri Bottas": 7.00, 
    "Guanyu Zhou": 16.00, 
    "Daniel Ricciardo": 9.00, 
    "Yuki Tsunoda": 13.00,
    "Lance Stroll": 14.50
}

average_finish_position = {
    "Max Verstappen": 4.80,
    "Sergio Pérez": 6.50,
    "Charles Leclerc": 5.50,
    "Carlos Sainz": 6.20,
    "Lewis Hamilton": 5.00,
    "George Russell": 9.00,
    "Fernando Alonso": 7.50,
    "Lando Norris": 5.80,
    "Oscar Piastri": 8.50,
    "Pierre Gasly": 10.50,
    "Esteban Ocon": 11.20,
    "Kevin Magnussen": 13.00,
    "Nico Hülkenberg": 11.50,
    "Alex Albon": 10.80,
    "Logan Sargeant": 19.00,
    "Valtteri Bottas": 7.50,
    "Guanyu Zhou": 17.50,
    "Daniel Ricciardo": 9.80,
    "Yuki Tsunoda": 14.00,
    "Lance Stroll": 15.00
}

# Définir les circuits disponibles à partir du dictionnaire de mapping
circuit_mapping = {
    "Bahreïn (Sakhir)": 1098,
    "Australie (Melbourne)": 1099,
    "Chine (Shanghai)": 1100,
    "Japon (Suzuka)": 1101,
    "Arabie saoudite (Djeddah)": 1102,
    "Miami (Floride)": 1103,
    "Emilie Romagne (Imola – Italie)": 1104,
    "Monaco (Monte-Carlo)": 1105,
    "Espagne (Barcelone)": 1106,
    "Canada (Montréal)": 1107,
    "Autriche (Spielberg)": 1108,
    "Royaume-Uni (Silverstone)": 1109,
    "Hongrie (Budapest)": 1110,
    "Grand Prix de Belgique (Spa-Francorchamps)": 1111,
    "Pays-Bas (Zandvoort)": 1112,
    "Italie (Monza)": 1113,
    "Azerbaïdjan (Bakou)": 1114,
    "Singapour (Marina Bay)": 1115,
    "USA (Austin)": 1116,
    "Mexique (Mexico City)": 1117,
    "Brésil (São Paulo)": 1118,
    "Las Vegas (Nevada)": 1119,
    "Qatar (Doha)": 1120,
    "Abu Dhabi (Yas Marina)": 1121
}

# Extraire les noms des circuits pour le sélecteur
available_circuits = list(circuit_mapping.keys())

# Simuler la course avec influence des moyennes et probabilités basées sur les notes
def simulate_race_with_probability(race_id, weather_df, circuits_df):
    # Mapper les 'driverid' aux noms des pilotes
    temps_par_courses_df['Nom du pilote'] = temps_par_courses_df['driverid'].map(driver_mapping)
    
    race_results = temps_par_courses_df[temps_par_courses_df['race_id'] == race_id]
    
    # Créer un DataFrame avec les noms des pilotes, leurs ratings, et leurs moyennes de départ/arrivée
    race_data = pd.DataFrame({
        'Nom du pilote': list(general_ratings.keys()),
        'Rating': list(general_ratings.values()),
        'Average_start': [average_starting_positions[pilote] for pilote in general_ratings.keys()],
        'Average_finish': [average_finish_position[pilote] for pilote in general_ratings.keys()]
    })
    
    # Ajouter une probabilité basée sur le rating du pilote pour influencer les résultats
    race_data['Position de départ'] = race_data['Average_start'] * (100 - race_data['Rating']) / 100 + np.random.uniform(0.9, 1.1, len(race_data))
    race_data['Position finale'] = race_data['Average_finish'] * (100 - race_data['Rating']) / 100 + np.random.uniform(0.9, 1.1, len(race_data))
    
    # Trier les résultats par position de départ et de finale
    race_data = race_data.sort_values('Position de départ').reset_index(drop=True)
    race_data['Position de départ'] = range(1, len(race_data) + 1)
    
    race_data = race_data.sort_values('Position finale').reset_index(drop=True)
    race_data['Position finale'] = range(1, len(race_data) + 1)

    return race_data[['Nom du pilote', 'Position de départ', 'Position finale']]

# Générer un graphique dynamique de la course
def generate_dynamic_race_graph(simulated_race_results):
    fig = go.Figure()

    race_progress = np.linspace(0, 100, 10)
    
    for i, row in simulated_race_results.iterrows():
        start_position = row['Position de départ']
        end_position = row['Position finale']
        
        initial_position = np.linspace(start_position, end_position, len(race_progress))
        position_progress = initial_position
        
        fig.add_trace(go.Scatter(
            x=race_progress,
            y=position_progress,
            mode='lines+markers',
            name=row['Nom du pilote']
        ))

    fig.update_layout(
        title="Simulation dynamique de la course avec dépassements",
        xaxis_title="Progression de la course (%)",
        yaxis_title="Position",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 21)),
            autorange="reversed"
        ),
        xaxis=dict(
            range=[0, 100]
        ),
        showlegend=True
    )

    return fig

# Interface utilisateur avec Streamlit
st.title('Simulation de Course F1 avec Influence des Notes Générales et Moyennes Historiques')

# Sélectionner un circuit
selected_circuit = st.selectbox("Choisissez un circuit", available_circuits)

# Récupérer le race_id correspondant au circuit
selected_race_id = circuit_mapping[selected_circuit]

# Simuler la course lorsque le bouton est cliqué
if st.button("Simuler la course"):
    simulated_race_results = simulate_race_with_probability(selected_race_id, weather_df, circuits_df)
    
    # Afficher les résultats avec la position de départ et la position finale
    st.subheader(f"Résultats simulés de la course : {selected_circuit}")
    st.dataframe(simulated_race_results[['Nom du pilote', 'Position de départ', 'Position finale']])
    
    # Générer le graphique dynamique des dépassements
    race_graph = generate_dynamic_race_graph(simulated_race_results)
    
    # Afficher le graphique
    st.plotly_chart(race_graph)

