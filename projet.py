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

def filter_data_for_seasons(races_df, results_df, seasons_df, years):
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

    # Simuler les courses pour les années spécifiées
    simulated_races_list = []
    for year in years:
        num_races = 23  # Nombre typique de courses dans une saison de F1
        simulated_races = pd.DataFrame({
            'raceid': range(max(races_df['raceid'].max() + 1, 1), max(races_df['raceid'].max() + 1, 1) + num_races),
            'year': [year] * num_races,
            'round': range(1, num_races + 1),
            'circuitid': range(1, num_races + 1),
            'name': [f"Race {i}" for i in range(1, num_races + 1)],
            'date': pd.date_range(start=f'{year}-01-01', periods=num_races, freq='W'),
            'time': ['15:00:00'] * num_races,
            'url': [f"http://en.wikipedia.org/wiki/{year}_Race_{i}" for i in range(1, num_races + 1)],
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
        simulated_races_list.append(simulated_races)

    # Ajouter les courses simulées à races_df
    races_df_simulated = pd.concat([races_df] + simulated_races_list, ignore_index=True)

    # Filtrage des données pour les années spécifiées
    seasons_filtered = seasons_df[seasons_df['year'].isin(years)]
    if seasons_filtered.empty:
        st.warning(f"Aucune donnée pour les saisons {years} trouvée dans seasons_df.")
        return None, None

    races_filtered = races_df_simulated[races_df_simulated['year'].isin(years)]
    results_filtered = results_df[results_df['raceid'].isin(races_filtered['raceid'])]

    return races_filtered, results_filtered

def main():
    races_df, results_df, seasons_df = load_data()

    # Vérifie chaque DataFrame individuellement
    if any(df is None for df in [races_df, results_df, seasons_df]):
        st.error("Une ou plusieurs données n'ont pas été chargées correctement.")
        return

    # Filtrage des données pour 2023 et 2024
    years = [2023, 2024]
    races_filtered, results_filtered = filter_data_for_seasons(races_df, results_df, seasons_df, years)

    # Affichage des données filtrées
    if races_filtered is not None and results_filtered is not None:
        st.write("Courses en 2023 et 2024:")
        st.dataframe(races_filtered)

        st.write("Résultats en 2023 et 2024:")
        st.dataframe(results_filtered)

if __name__ == "__main__":
    main()
