import pandas as pd
import streamlit as st

def load_data():
    try:
        races_df = pd.read_csv('races.csv')
        results_df = pd.read_csv('results.csv')
        seasons_df = pd.read_csv('seasons.csv')
        drivers_df = pd.read_csv('drivers.csv')
        constructors_df = pd.read_csv('constructors.csv')
        
        st.success("Tous les fichiers ont été chargés avec succès.")
        st.write("Colonnes de races_df:", races_df.columns)
        st.write("Colonnes de results_df:", results_df.columns)
        st.write("Colonnes de seasons_df:", seasons_df.columns)
        st.write("Colonnes de drivers_df:", drivers_df.columns)
        st.write("Colonnes de constructors_df:", constructors_df.columns)
        
        return races_df, results_df, seasons_df, drivers_df, constructors_df

    except FileNotFoundError as e:
        st.error(f"Erreur lors du chargement des fichiers : {e}")
        return None, None, None, None, None

def filter_data_for_2024(races_df, results_df, seasons_df):
    if races_df is None or results_df is None or seasons_df is None:
        st.error("Données manquantes pour le filtrage.")
        return None, None
    
    if 'seasonId' not in seasons_df.columns or 'season' not in seasons_df.columns:
        st.error("Colonnes manquantes dans seasons_df.")
        return None, None

    if 'raceId' not in races_df.columns or 'seasonId' not in races_df.columns:
        st.error("Colonnes manquantes dans races_df.")
        return None, None

    if 'resultId' not in results_df.columns or 'raceId' not in results_df.columns:
        st.error("Colonnes manquantes dans results_df.")
        return None, None

    season_2024_id = seasons_df[seasons_df['season'] == 2024]['seasonId']
    if season_2024_id.empty:
        st.warning("Aucune donnée pour la saison 2024 trouvée dans seasons_df.")
        return None, None

    season_2024_id = season_2024_id.iloc[0]

    races_2024 = races_df[races_df['seasonId'] == season_2024_id]
    results_2024 = results_df[results_df['raceId'].isin(races_2024['raceId'])]

    return races_2024, results_2024

def main():
    races_df, results_df, seasons_df, drivers_df, constructors_df = load_data()

    # Vérifie chaque DataFrame individuellement
    if any(df is None for df in [races_df, results_df, seasons_df, drivers_df, constructors_df]):
        st.error("Une ou plusieurs données n'ont pas été chargées correctement.")
        return

    races_2024, results_2024 = filter_data_for_2024(races_df, results_df, seasons_df)

    if races_2024 is not None and results_2024 is not None:
        st.write("Courses en 2024:", races_2024)
        st.write("Résultats en 2024:", results_2024)

if __name__ == "__main__":
    main()
