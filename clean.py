import pandas as pd

def clean_lap_times(file_path):
    # Lire le fichier CSV
    df = pd.read_csv(file_path)
    
    # Afficher les premières lignes et les noms des colonnes du DataFrame pour inspection
    print("Colonnes disponibles :", df.columns)
    print(df.head())
    
    # Exemple de nettoyage (à adapter selon les besoins spécifiques)
    # 1. Supprimer les colonnes inutiles
    columns_to_keep = ['raceId', 'driverId', 'lap', 'time']  # Garder uniquement les colonnes nécessaires
    df = df[columns_to_keep]
    
    # 2. Traiter les valeurs manquantes (exemple : supprimer les lignes avec des valeurs manquantes)
    df = df.dropna()
    
    # 3. Convertir les types de données
    # Convertir 'lap' en entier
    df['lap'] = df['lap'].astype(int)
    
    # Fonction pour convertir le temps de 'minute:seconde.milliseconde' en millisecondes
    def time_to_ms(time_str):
        try:
            # Assurer que le temps suit le format attendu
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = parts
                    seconds, milliseconds = seconds.split('.')
                    return int(minutes) * 60000 + int(seconds) * 1000 + int(milliseconds)
                else:
                    raise ValueError("Format de temps inattendu.")
            else:
                raise ValueError("Format de temps inattendu.")
        except Exception as e:
            print(f"Erreur lors de la conversion de '{time_str}': {e}")
            return None  # Retourner None pour les valeurs non valides
    
    # Appliquer la fonction de conversion et gérer les erreurs
    df['time'] = df['time'].apply(time_to_ms)
    
    # Supprimer les lignes avec des valeurs non convertibles
    df = df.dropna(subset=['time'])
    
    # 4. Normaliser les valeurs
    # Convertir les noms de colonnes en minuscules
    df.columns = [col.lower() for col in df.columns]
    
    return df

# Nettoyer le fichier 'lap_times.csv' et sauvegarder le résultat
lap_times_df = clean_lap_times('lap_times.csv')
lap_times_df.to_csv('cleaned_lap_times.csv', index=False)
print('Nettoyage terminé pour lap_times.csv.')
