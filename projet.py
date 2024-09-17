import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def load_data():
    try:
        races_df = pd.read_csv('cleaned_races.csv')
        results_df = pd.read_csv('cleaned_results.csv')
        seasons_df = pd.read_csv('cleaned_seasons.csv')
        constructors_df = pd.read_csv('cleaned_constructors.csv')
        drivers_df = pd.read_csv('cleaned_drivers.csv')
        return races_df, results_df, seasons_df, constructors_df, drivers_df
    except FileNotFoundError as e:
        st.error(f"Erreur lors du chargement des fichiers : {e}")
        return None, None, None, None, None

def filter_data_for_seasons(races_df, results_df, seasons_df, constructors_df, drivers_df, years):
    if races_df is None or results_df is None or seasons_df is None:
        st.error("Données manquantes pour le filtrage.")
        return None, None

    missing_columns = []
    if 'year' not in seasons_df.columns:
        missing_columns.append('year')

    if missing_columns:
        st.error(f"Colonnes manquantes dans seasons_df: {', '.join(missing_columns)}")
        return None, None

    if 'raceid' not in races_df.columns or 'year' not in races_df.columns:
        st.error("Colonnes manquantes dans races_df.")
        return None, None

    if 'resultid' not in results_df.columns or 'raceid' not in results_df.columns:
        st.error("Colonnes manquantes dans results_df.")
        return None, None

    # Fusionner les DataFrames pour obtenir les noms des pilotes et des constructeurs
    results_with_names = results_df.merge(drivers_df[['driverid', 'forename', 'surname']],
                                          left_on='driverid', right_on='driverid', how='left')
    results_with_names = results_with_names.merge(constructors_df[['constructorid', 'name']],
                                                  left_on='constructorid', right_on='constructorid', how='left')

    # Renommer les colonnes pour la création des graphiques
    results_with_names = results_with_names.rename(columns={
        'name': 'name_constructor',
        'forename': 'driver_forename',
        'surname': 'driver_surname'
    })

    simulated_races_list = []
    for year in years:
        num_races = 23
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

    races_df_simulated = pd.concat([races_df] + simulated_races_list, ignore_index=True)

    seasons_filtered = seasons_df[seasons_df['year'].isin(years)]
    if seasons_filtered.empty:
        st.warning(f"Aucune donnée pour les saisons {years} trouvée dans seasons_df.")
        return None, None

    races_filtered = races_df_simulated[races_df_simulated['year'].isin(years)]
    results_filtered = results_with_names[results_with_names['raceid'].isin(races_filtered['raceid'])]

    return races_filtered, results_filtered

def create_plots(races_df, results_df):
    results_by_constructor = results_df.groupby('name_constructor')['points'].sum().reset_index()
    fig_constructor_points = px.bar(
        results_by_constructor,
        x='name_constructor',
        y='points',
        title='Points Totaux par Constructeur (2023 et 2024)',
        labels={'name_constructor': 'Nom du Constructeur', 'points': 'Points'},
        text='points'
    )
    fig_constructor_points.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    results_by_driver = results_df.groupby(['driver_forename', 'driver_surname'])['points'].sum().reset_index()
    fig_driver_points = px.bar(
        results_by_driver,
        x='driver_forename',
        y='points',
        title='Points Totaux par Pilote (2023 et 2024)',
        labels={'driver_forename': 'Prénom du Pilote', 'points': 'Points'},
        text='points'
    )
    fig_driver_points.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig_constructor_points, fig_driver_points

def main():
    races_df, results_df, seasons_df, constructors_df, drivers_df = load_data()

    if any(df is None for df in [races_df, results_df, seasons_df, constructors_df, drivers_df]):
        st.error("Une ou plusieurs données n'ont pas été chargées correctement.")
        return

    years = [2023, 2024]
    races_filtered, results_filtered = filter_data_for_seasons(races_df, results_df, seasons_df, constructors_df, drivers_df, years)

    if races_filtered is not None and results_filtered is not None:
        st.write("Courses en 2023 et 2024:")
        st.dataframe(races_filtered)

        st.write("Résultats en 2023 et 2024:")
        st.dataframe(results_filtered)

        fig_constructor_points, fig_driver_points = create_plots(races_filtered, results_filtered)

        st.plotly_chart(fig_constructor_points)
        st.plotly_chart(fig_driver_points)

if __name__ == "__main__":
    main()
