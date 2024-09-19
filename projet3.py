import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Charger et afficher le logo F1 avec une taille réduite
logo_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/F1-LOGO.png'  # Mets ici le bon chemin vers le logo
logo = Image.open(logo_path)
st.image(logo, width=300, caption="Simulation Formule 1")  # Modification du titre

# Charger les fichiers CSV avec le chemin correct sur ta machine
temps_par_courses_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/temps_par_courses.csv'
drivers_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/drivers.csv'
qualifying_path = 'C:/Users/Marco Luis/Documents/PROJET Data prediction VVA MARCO LUIS/cleaned_qualifying.csv'

# Charger les données
temps_par_courses_df = pd.read_csv(temps_par_courses_path)
drivers_df = pd.read_csv(drivers_path)
qualifying_df = pd.read_csv(qualifying_path)

# S'assurer que la colonne 'milliseconds' est bien numérique
temps_par_courses_df['milliseconds'] = pd.to_numeric(temps_par_courses_df['milliseconds'], errors='coerce')

# Filtrer les valeurs non numériques (par exemple, les NaN)
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

# Liste des circuits et association avec race_id
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

# Récupérer les noms des circuits disponibles
available_circuits = list(circuit_mapping.keys())

# Fonction pour intégrer les positions de départ depuis le fichier des qualifications
def get_starting_positions(race_id):
    qualifying_results = qualifying_df[qualifying_df['raceid'] == race_id][['driverid', 'position']]
    qualifying_results = qualifying_results.rename(columns={'position': 'Position de départ'})
    qualifying_results = qualifying_results.drop_duplicates(subset='Position de départ')  # S'assurer qu'il n'y a pas de doublons
    return qualifying_results

# Fonction pour simuler une course avec variation aléatoire et dépassements
def simulate_race_with_realistic_results(race_id):
    np.random.seed()  # Utiliser un seed différent pour garantir des résultats variés à chaque simulation
    
    # Filtrer les résultats de la course sélectionnée
    race_results = temps_par_courses_df[temps_par_courses_df['race_id'] == race_id]
    
    # Calculer les temps moyens pour chaque pilote
    pilot_avg_times = race_results.groupby('driverid')['milliseconds'].mean().reset_index()
    
    # Inclure tous les pilotes même s'ils ne participaient pas à la course
    full_driver_list = pd.DataFrame(list(driver_mapping.items()), columns=['driverid', 'Nom du pilote'])
    race_results = pd.merge(full_driver_list, pilot_avg_times, on='driverid', how='left')

    # Intégrer les positions de départ depuis les qualifications
    starting_positions = get_starting_positions(race_id)
    race_results = pd.merge(race_results, starting_positions, on='driverid', how='left')

    # Gérer les positions manquantes : assigner les places restantes de manière ordonnée
    missing_positions = race_results['Position de départ'].isnull()
    available_positions = set(range(1, 21)) - set(race_results['Position de départ'].dropna().astype(int))
    race_results.loc[missing_positions, 'Position de départ'] = list(available_positions)[:missing_positions.sum()]

    # Générer des variations autour des temps historiques pour la simulation
    variation_factor = np.random.uniform(0.90, 1.10, len(race_results))
    race_results['simulated_time'] = race_results['milliseconds'] * variation_factor
    
    # Trier par temps simulé pour obtenir le classement final
    race_results = race_results.sort_values('simulated_time')

    # Générer un classement basé sur les temps simulés
    race_results['Position finale'] = range(1, len(race_results) + 1)
    
    return race_results[['Nom du pilote', 'Position de départ', 'Position finale', 'simulated_time']]

# Fonction pour générer un graphique dynamique avec dépassements
def generate_dynamic_race_graph(simulated_race_results):
    fig = go.Figure()

    # Simulation des positions à différents moments de la course (10 moments)
    race_progress = np.linspace(0, 100, 10)  # Progression de 0% à 100%
    num_drivers = len(simulated_race_results)
    
    # Simuler des dépassements pendant la course
    for i, row in simulated_race_results.iterrows():
        start_position = row['Position de départ']  # Position de départ
        end_position = row['Position finale']  # Position finale simulée
        
        # Simuler la progression dynamique des positions
        initial_position = np.linspace(start_position, end_position, len(race_progress))
        position_progress = initial_position  # Le progrès est linéaire entre départ et arrivée
        
        fig.add_trace(go.Scatter(
            x=race_progress,
            y=position_progress,
            mode='lines+markers',
            name=row['Nom du pilote']
        ))

    # Mise à jour du layout
    fig.update_layout(
        title="Simulation dynamique de la course avec dépassements",
        xaxis_title="Progression de la course (%)",
        yaxis_title="Position",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 21)),  # Positions de 1 à 20
            autorange="reversed"  # Le premier est en haut
        ),
        xaxis=dict(
            range=[0, 100]  # Progression de 0 à 100%
        ),
        showlegend=True
    )
    
    return fig

# Interface utilisateur avec Streamlit
st.title('Simulation de Course F1 avec Dépassements Dynamiques')

# Sélectionner un circuit
selected_circuit = st.selectbox("Choisissez un circuit", available_circuits)

# Récupérer le race_id correspondant au circuit
selected_race_id = circuit_mapping[selected_circuit]

# Simuler la course lorsque le bouton est cliqué
if st.button("Simuler la course"):
    simulated_race_results = simulate_race_with_realistic_results(selected_race_id)
    
    # Afficher les résultats avec la position de départ et la position finale
    st.subheader(f"Résultats simulés de la course : {selected_circuit}")
    st.dataframe(simulated_race_results[['Position de départ', 'Nom du pilote', 'Position finale']])
    
    # Générer le graphique dynamique des dépassements
    race_graph = generate_dynamic_race_graph(simulated_race_results)
    
    # Afficher le graphique
    st.plotly_chart(race_graph)
