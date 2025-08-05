import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURARE
# ==============================================================================
FILES_TO_SCORE = [
    "rezultate_inainte_de_scantailor.csv",
    "rezultate_dupa_scantailor.csv"
]

SCORING_WEIGHTS = {
    'abs_skew':     0.50, # 50%
    'top_margin':   0.20, # 20%
    'outer_margin': 0.30  # 30%
}

# --- NOU: ZONE DE TOLERANȚĂ ---
# Aici definim ce înseamnă "perfect". Orice valoare în acest interval
# în jurul medianei va primi o penalizare de ZERO.
TOLERANCES = {
    # Pentru skew, folosim o valoare absolută (în grade).
    # O toleranță de 0.5 înseamnă că orice skew între (mediana - 0.5) și (mediana + 0.5) e perfect.
    'abs_skew_absolute': 0.5, 

    # Pentru margini, folosim un procent.
    # O toleranță de 5.0 înseamnă că orice margine în +/- 5% din mediană e perfectă.
    'top_margin_percent': 5.0,
    'outer_margin_percent': 5.0
}

# Date despre imagini (confirmate din rulările anterioare)
IMG_WIDTH = 2915
IMG_HEIGHT = 4592
START_PAGE_IS_RECTO = True
# ==============================================================================
# SFÂRȘIT CONFIGURARE
# ==============================================================================

def calculate_intelligent_score(row, median_values, weights, tolerances):
    """
    Calculează scorul ponderat, luând în considerare zonele de toleranță.
    """
    weighted_error_sum = 0.0
    
    for metric, weight in weights.items():
        current_value = row[metric]
        median_value = median_values[metric]
        
        # Determinăm toleranța pentru metrica curentă
        if metric == 'abs_skew':
            tolerance_value = tolerances.get('abs_skew_absolute', 0)
        else:
            tolerance_percent = tolerances.get(f'{metric}_percent', 0)
            tolerance_value = median_value * (tolerance_percent / 100.0)

        # Calculăm limita superioară și inferioară a zonei de perfecțiune
        lower_bound = median_value - tolerance_value
        upper_bound = median_value + tolerance_value

        deviation = 0.0
        # Penalizăm doar dacă valoarea este ÎN AFARA zonei de toleranță
        if current_value < lower_bound:
            deviation = lower_bound - current_value
        elif current_value > upper_bound:
            deviation = current_value - upper_bound
        
        # Calculăm abaterea procentuală pe baza deviației REALE (nu cea de la mediană)
        if median_value == 0:
            deviation_percent = deviation * 100 # Scalare arbitrară
        else:
            deviation_percent = (deviation / median_value) * 100
            
        weighted_error_sum += (deviation_percent / 100.0) * weight
    
    score = max(0, 100 - (weighted_error_sum * 100))
    return score

def score_csv_file(filepath):
    """
    Funcția principală care orchestrează analiza unui fișier CSV.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"EROARE: Fișierul '{filepath}' nu a fost găsit.")
        return

    print("\n" + "=" * 80)
    print(f"Analiză de Calitate pentru fișierul: '{filepath}'")
    print("=" * 80)
    
    # --- Calculăm coloanele de margini necesare ---
    df['page_num'] = df['filename'].str.extract(r'(\d+)$').astype(int)
    df['top_margin'] = df['y1']
    is_recto_condition = ((df['page_num'] - 1) % 2 == 0) if START_PAGE_IS_RECTO else (df['page_num'] % 2 == 0)
    df['outer_margin'] = np.where(is_recto_condition, IMG_WIDTH - df['x2'], df['x1'])
    # --- Sfârșit calcul margini ---

    required_columns = list(SCORING_WEIGHTS.keys())
    median_values = df[required_columns].median()
    
    print("Profilul Ideal (Medianele) pentru acest set de date:")
    print(median_values.round(2))

    df['score'] = df.apply(lambda row: calculate_intelligent_score(row, median_values, SCORING_WEIGHTS, TOLERANCES), axis=1)

    average_score = df['score'].mean()
    
    print("\n--- REZULTAT FINAL ---")
    print(f"{'SCOR MEDIU DE CONFORMITATE (PONDERAT)':<45}: {average_score:.2f} / 100.00")
    print(f"Metrici considerate: {', '.join([f'{k} ({v*100:.0f}%)' for k, v in SCORING_WEIGHTS.items()])}")

    df_sorted = df.sort_values(by='score', ascending=True)
    
    print("\nTop 5 pagini cu cel mai MIC scor (cele mai mari abateri):")
    print(df_sorted[['filename', 'score'] + required_columns].head().round(2))

    print("\nTop 5 pagini cu cel mai MARE scor (cele mai apropiate de ideal):")
    print(df_sorted[['filename', 'score'] + required_columns].tail().round(2))
    print("-" * 80)


if __name__ == "__main__":
    for csv_file in FILES_TO_SCORE:
        score_csv_file(csv_file)