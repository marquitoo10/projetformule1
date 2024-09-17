import pandas as pd
import numpy as np
import streamlit as st

def load_data():
    try:
        races_df = pd.read_csv('cleaned_races.csv')
        results_df = pd.read_csv('cleaned_results.csv')
        seasons_df = pd.read_csv('cleaned_seasons.csv')
        return races_df, results_df, seasons_df
    except FileNotFoundError as e:
        st.error(f"Erreur lors du chargement des fichiers : {e}")
        return None, None, None

def filter_data_for_2024(races_df, results_df, seasons_df):
    if races_df is None or results_df is None or seasons_df is None:
        st.error("Données manquantes pour le filtrage.")
        return None, None
    
    # Vérification des colonnes dans seasons_df
    missing_columns = []
    if 'year' not in seasons_df.columns:
        missing_columns.append('year')

    if missing_columns:
        st.error(f"Colonnes manquantes dans seasons_df: {', '.join(missing_columns)}")
        return None, None

    # Vérification des colonnes dans races_df et results_df
    if 'raceid' not in races_df.columns or 'year' not in races_df.columns:
        st.error("Colonnes manquantes dans races_df.")
        return None, None

    if 'resultid' not in results_df.columns or 'raceid' not in results_df.columns:
        st.error("Colonnes manquantes dans results_df.")
        return None, None

    # Simuler les courses pour 2024
    num_races = 23  # Nombre typique de courses dans une saison de F1
    simulated_races = pd.DataFrame({
        'raceid': range(max(races_df['raceid'].max() + 1, 1), max(races_df['raceid'].max() + 1, 1) + num_races),
        'year': [2024] * num_races,
        'round': range(1, num_races + 1),
        'circuitid': range(1, num_races + 1),
        'name': [f"Race {i}" for i in range(1, num_races + 1)],
        'date': pd.date_range(start='2024-01-01', periods=num_races, freq='W'),
        'time': ['15:00:00'] * num_races,
        'url': [f"http://en.wikipedia.org/wiki/2024_Race_{i}" for i in range(1, num_races + 1)],
        'fp1_date': [np.nan] * num_races,
        'fp1_time': [np.nan] * num_races,
        'fp2_date': [np.nan] * num_races,
        'fp2_time': [np.nan] * num_races,
        'fp3_date': [np.nan] * num_races,
        'fp3_time': [np.nan] * num_races,
        'quali_date': [np.nan] * num_races,
        'quali_time': [np.nan] * num_races,
        'sprint_date': [np.nan] * num_races,
        'sprint_time': [np.nan] * num_races
    })

    # Ajouter les courses simulées à races_df
    races_df_simulated = pd.concat([races_df, simulated_races], ignore_index=True)

    # Filtrage des données pour la saison 2024
    season_2024 = seasons_df[seasons_df['year'] == 2024]
    if season_2024.empty:
        st.warning("Aucune donnée pour la saison 2024 trouvée dans seasons_df.")
        return None, None

    # En supposant que les courses de 2024 sont identifiées dans races_df par la colonne 'year'
    races_2024 = races_df_simulated[races_df_simulated['year'] == 2024]
    results_2024 = results_df[results_df['raceid'].isin(races_2024['raceid'])]

    return races_2024, results_2024

def main():
    races_df, results_df, seasons_df = load_data()

    # Vérifie chaque DataFrame individuellement
    if any(df is None for df in [races_df, results_df, seasons_df]):
        st.error("Une ou plusieurs données n'ont pas été chargées correctement.")
        return

    # Filtrage des données pour 2024
    races_2024, results_2024 = filter_data_for_2024(races_df, results_df, seasons_df)

    # Affichage des données filtrées
    if races_2024 is not None and results_2024 is not None:
        st.write("Courses en 2024:")
        st.dataframe(races_2024)

        st.write("Résultats en 2024:")
        st.dataframe(results_2024)

if __name__ == "__main__":
    main()
