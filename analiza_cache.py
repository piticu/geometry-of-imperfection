import pandas as pd
from pathlib import Path
import os

# ==============================================================================
# CONFIGURARE
# ==============================================================================
# Directorul de unde se citesc datele de cache
CACHE_DIRECTORY = "cache_google_vision"

# Numele fișierului CSV în care se vor salva rezultatele
OUTPUT_CSV_FILE = "rezultate_inainte_de_scantailor.csv"
# ==============================================================================
# SFÂRȘIT CONFIGURARE
# ==============================================================================

def analyze_cache_folder():
    """
    Scanează directorul de cache, extrage datele din fiecare fișier .txt,
    le încarcă într-un DataFrame pandas și îl salvează ca fișier CSV.
    """
    cache_path = Path(CACHE_DIRECTORY)
    if not cache_path.is_dir():
        print(f"EROARE: Directorul de cache '{CACHE_DIRECTORY}' nu a fost găsit.")
        print("Asigurați-vă că rulați acest script în același folder ca și scriptul principal.")
        return

    # Găsim toate fișierele .txt din directorul de cache
    cache_files = list(cache_path.glob('*.txt'))

    if not cache_files:
        print(f"EROARE: Nu s-au găsit fișiere .txt în directorul '{CACHE_DIRECTORY}'.")
        return

    print(f"--- Analiză Cache ---")
    print(f"S-au găsit {len(cache_files)} fișiere de cache în '{CACHE_DIRECTORY}'.")

    all_pages_data = []

    # Iterăm prin fiecare fișier de cache
    for file_path in cache_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                print(f"  -> Avertisment: Fișierul cache '{file_path.name}' este gol. Va fi ignorat.")
                continue

            parts = content.split(',')
            if len(parts) == 5:
                # Extragem datele și le convertim la tipul corect
                x1 = int(parts[0])
                y1 = int(parts[1])
                x2 = int(parts[2])
                y2 = int(parts[3])
                skew = float(parts[4])

                # Adăugăm datele extrase în lista noastră
                all_pages_data.append({
                    'filename': file_path.stem, # Numele fișierului fără extensie
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'skew': skew
                })
            else:
                print(f"  -> Avertisment: Fișierul cache '{file_path.name}' are un format invalid. Va fi ignorat.")
        
        except (ValueError, IndexError) as e:
            print(f"  -> Avertisment: Eroare la procesarea fișierului '{file_path.name}': {e}. Va fi ignorat.")

    if not all_pages_data:
        print("Nu s-au putut extrage date valide din niciun fișier de cache.")
        return

    # Creăm DataFrame-ul pandas
    df = pd.DataFrame(all_pages_data)

    # Adăugăm coloane suplimentare pentru o analiză mai ușoară
    print("\nSe adaugă coloane calculate (width, height, abs_skew)...")
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['abs_skew'] = abs(df['skew'])

    # Sortăm DataFrame-ul după numele fișierului pentru consistență
    df = df.sort_values(by='filename').reset_index(drop=True)

    try:
        # Salvăm DataFrame-ul în fișierul CSV
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nSUCCES: Datele au fost salvate cu succes în fișierul '{OUTPUT_CSV_FILE}'.")
    except Exception as e:
        print(f"\nEROARE: Nu s-a putut salva fișierul CSV: {e}")
        return

    # Afișăm un sumar al datelor
    print("\n--- Sumar Date Extrase ---")
    print("Primele 5 rânduri din tabel:")
    print(df.head())
    
    print("\nDescriere statistică a datelor numerice:")
    # Folosim .round(2) pentru a afișa numerele mai frumos
    print(df.describe().round(2))


if __name__ == "__main__":
    analyze_cache_folder()