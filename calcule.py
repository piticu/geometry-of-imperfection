# --- START OF FILE calcule.py (cu vizualizare oglindită, cache, throttling, pre-filtrare și diagnostic) ---
import cv2
import numpy as np
import os
from pathlib import Path
import math
import io
import time

from google.cloud import vision
# Inițializăm clientul Google Vision o singură dată pentru performanță
GOOGLE_VISION_CLIENT = vision.ImageAnnotatorClient()

# ==============================================================================
# CONFIGURARE PRINCIPALĂ (modifică aceste valori după necesități)
# ==============================================================================
# Parametri generali
DPI = 600                   # Rezoluția (DPI) a imaginilor scanate
DIRECTORY = "./scans"       # Directorul unde se află imaginile
FILE_TYPE = 'webp'          # Tipul fișierelor de procesat ('tif', 'jpg', 'png', etc.)
START_PAGE_IS_RECTO = True  # Prima pagină este o pagină dreapta (recto)? (True/False)

# Parametri algoritm
ALGORITHM = 'google_vision' # Algoritmul de detecție folosit
TARGET_DPI_FOR_API = 150    # Rezoluția la care se redimensionează imaginile pentru API
WEBP_COMPRESSION = 75       # Calitatea compresiei WebP (0-100, 100=lossless)

# Parametri Throttling (limitare apeluri API)
THROTTLING_ENABLED = True   # Activează pauza între apelurile API (True/False)
API_CALL_DELAY_S = 2        # Pauza în secunde între apelurile care nu sunt în cache

# Parametri Cache
CACHE_ENABLED = True        # Activează sistemul de cache (True/False)
CACHE_DIRECTORY = f"cache_{ALGORITHM}" # Directorul pentru stocarea rezultatelor cache

# Parametri de Pre-Filtrare Semantică
# ATENȚIE: Aceste valori sunt probabil prea agresive. Scriptul se va opri și vă va arăta valorile reale.
PREFILTER_ENABLED = False     # Activează/dezactivează pre-filtrarea
MIN_BLOCK_HEIGHT_PERC = 75.0  # Blocul de text trebuie să ocupe minim acest % din înălțimea paginii
MIN_BLOCK_WIDTH_PERC = 70.0   # Blocul de text trebuie să ocupe minim acest % din lățimea paginii
MAX_SKEW_DEGREES = 5.0        # Abaterea maximă permisă a unghiului în grade

# Parametri Debug și Vizualizare
DEBUG = False                   # Activează generarea imaginilor de debug (True/False)
VISUALIZATION_MODE = 'inliers'  # Alege modul de vizualizare: 'all', 'inliers', 'outliers'
# ==============================================================================
# SFÂRȘIT CONFIGURARE
# ==============================================================================


# ==============================================================================
# DEFINIȚIILE FUNCȚIILOR (Neschimbate)
# ==============================================================================

def detect_with_google_vision(image_path, target_dpi=300, original_dpi=600):
    print(f"  -> Procesare cu Google Vision (API): {Path(image_path).name}")
    try:
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print(f"  -> Eroare: Nu am putut încărca imaginea {Path(image_path).name}")
            return None, 0.0
        
        original_h, original_w = original_img.shape[:2]
        scale_factor = target_dpi / original_dpi
        resized_w = int(original_w * scale_factor)
        resized_h = int(original_h * scale_factor)
        
        resized_img = cv2.resize(original_img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        params = [cv2.IMWRITE_WEBP_QUALITY, WEBP_COMPRESSION]
        success, encoded_image = cv2.imencode('.webp', resized_img, params)
        if not success:
            raise Exception("Nu am putut encoda imaginea redimensionată în format WebP.")
        
        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        
        response = GOOGLE_VISION_CLIENT.document_text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)
            
        annotation = response.full_text_annotation

        if not annotation.pages or not annotation.pages[0].blocks:
            print(f"  -> Avertisment: Google Vision nu a detectat text în {Path(image_path).name}")
            return None, 0.0

        all_vertices = [v for page in annotation.pages for block in page.blocks for v in block.bounding_box.vertices]
        
        if not all_vertices:
            return None, 0.0
            
        min_x_resized = min(v.x for v in all_vertices)
        min_y_resized = min(v.y for v in all_vertices)
        max_x_resized = max(v.x for v in all_vertices)
        max_y_resized = max(v.y for v in all_vertices)

        inverse_scale_factor = original_dpi / target_dpi
        content_box = (
            int(min_x_resized * inverse_scale_factor),
            int(min_y_resized * inverse_scale_factor),
            int(max_x_resized * inverse_scale_factor),
            int(max_y_resized * inverse_scale_factor)
        )

        skew_angle = 0.0
        if annotation.pages[0].blocks:
            first_block_vertices = annotation.pages[0].blocks[0].bounding_box.vertices
            v0, v1 = first_block_vertices[0], first_block_vertices[1]
            angle_rad = math.atan2(v1.y - v0.y, v1.x - v0.x)
            skew_angle = math.degrees(angle_rad)

        return content_box, skew_angle

    except Exception as e:
        print(f"A apărut o eroare la apelul Google Vision API pentru {Path(image_path).name}: {e}")
        return None, 0.0

def compute_statistics(page_metrics):
    if not page_metrics: raise ValueError("Lista de metrici este goală.")
    arr = np.array(page_metrics)
    q1, q3 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
    iqr = q3 - q1; iqr[iqr == 0] = 1e-9
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
    mask = np.all((arr >= lower_bound) & (arr <= upper_bound), axis=1)
    
    filtered = arr[mask]
    if filtered.shape[0] == 0: filtered = arr 
    
    mins, maxs, medians = np.min(arr, axis=0), np.max(arr, axis=0), np.median(filtered, axis=0)
    metric_names = ['inner_margin', 'outer_margin', 'top_margin', 'bottom_margin', 'width', 'height', 'skew']
    stats = {}
    for i, name in enumerate(metric_names):
        if name == 'skew': stats[name] = (mins[i], medians[i], maxs[i])
        else: stats[name] = (int(mins[i]), int(medians[i]), int(maxs[i]))
    
    return stats, mask

def pixels_to_mm(pixels, dpi): return (pixels / dpi) * 25.4

def visualize_text_blocks(image_size, boxes, output_path, median_box=None):
    overlay = np.zeros(image_size, dtype=np.uint16)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        overlay[y1:y2, x1:x2] += 1
    overlay = np.log1p(overlay)
    if overlay.max() > 0: norm = np.uint8((overlay / overlay.max()) * 255)
    else: norm = np.uint8(overlay)
    
    color_overlay = cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
    
    if median_box:
        x1, y1, x2, y2 = map(int, median_box)
        cv2.rectangle(color_overlay, (x1, y1), (x2, y2), (255, 0, 0), 5)

    cv2.imwrite(output_path, color_overlay)

# ==============================================================================
# ZONA DE EXECUȚIE PRINCIPALĂ
# ==============================================================================

DEBUG_OUTPUT_DIR = f"debug_output_{ALGORITHM}"
output_overlay = f"suprapunere_contururi_{ALGORITHM}_{VISUALIZATION_MODE}.png"

if DEBUG: os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
if CACHE_ENABLED: os.makedirs(CACHE_DIRECTORY, exist_ok=True)

ftype_lower = FILE_TYPE.lower()
if ftype_lower == 'tif': image_extensions = ['.tif', '.tiff']
elif ftype_lower == 'jpg': image_extensions = ['.jpg', '.jpeg']
else: image_extensions = [f'.{ftype_lower}']

files = sorted([p for p in Path(DIRECTORY).glob("*.*") if p.suffix.lower() in image_extensions], key=lambda p: str(p.name))

if not files:
    print(f"Nu s-au găsit imagini ({', '.join(image_extensions)}) în directorul '{DIRECTORY}'.")
else:
    print(f"--- Procesare hibridă (Cache/API) pentru {len(files)} pagini ---")
    
    final_results = []
    for file in files:
        cache_path = Path(CACHE_DIRECTORY) / file.with_suffix('.txt').name
        box, skew_angle = None, None

        if CACHE_ENABLED and cache_path.exists():
            try:
                print(f"  -> Se încarcă rezultatul din cache: {cache_path.name}")
                with open(cache_path, 'r') as f: content = f.read().strip()
                parts = content.split(',')
                if len(parts) == 5:
                    box = tuple(int(p) for p in parts[:4])
                    skew_angle = float(parts[4])
            except (ValueError, IndexError):
                 print(f"  -> Avertisment: Cache corupt pentru {file.name}. Se re-procesează.")

        if box is None:
            if THROTTLING_ENABLED:
                print(f"  -> Throttling: Așteptare {API_CALL_DELAY_S} secundă(e)...")
                time.sleep(API_CALL_DELAY_S)
            
            box, skew_angle = detect_with_google_vision(file, target_dpi=TARGET_DPI_FOR_API, original_dpi=DPI)
            if box and CACHE_ENABLED:
                with open(cache_path, 'w') as f:
                    cache_line = f"{box[0]},{box[1]},{box[2]},{box[3]},{skew_angle}"
                    f.write(cache_line)
                print(f"  -> Rezultatul a fost salvat în cache: {cache_path.name}")
        
        if box:
            final_results.append({'filename': file.name, 'box': box, 'skew': skew_angle})
        else:
            print(f"Avertisment: Nu s-a detectat conținut în {file.name}")
    
    print(f"\n--- Detecție completă. Se calculează statisticile și se generează vizualizările. ---")
    
    if not final_results:
        print("Nu s-au putut încărca date. Programul se oprește.")
        exit()
    
    sample_img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    img_height, img_width = sample_img.shape
    
    valid_pages_indices = []
    if PREFILTER_ENABLED:
        print("\n--- Se aplică Pre-Filtrarea Semantică ---")
        min_h_px = img_height * (MIN_BLOCK_HEIGHT_PERC / 100.0)
        min_w_px = img_width * (MIN_BLOCK_WIDTH_PERC / 100.0)
        
        for i, res in enumerate(final_results):
            x1, y1, x2, y2 = res['box']
            box_w, box_h = x2 - x1, y2 - y1
            
            if (box_h >= min_h_px and box_w >= min_w_px and abs(res['skew']) <= MAX_SKEW_DEGREES):
                valid_pages_indices.append(i)
            else:
                print("-" * 60)
                print(f"DIAGNOSTIC: PRIMA PAGINĂ RESPINSĂ DE PRE-FILTRU (SCRIPT OPRIT)")
                print(f" -> Nume fișier: {res['filename']}")
                print(f" -> Motivul respingerii (cel puțin una din condiții e falsă):")
                cond_h = box_h >= min_h_px
                cond_w = box_w >= min_w_px
                cond_s = abs(res['skew']) <= MAX_SKEW_DEGREES
                print(f"    1. Înălțime OK? {cond_h} (Valoare: {box_h*100/img_height:.1f}%, Prag necesar: {MIN_BLOCK_HEIGHT_PERC}%)")
                print(f"    2. Lățime OK?   {cond_w} (Valoare: {box_w*100/img_width:.1f}%, Prag necesar: {MIN_BLOCK_WIDTH_PERC}%)")
                print(f"    3. Skew OK?     {cond_s} (Valoare: {abs(res['skew']):.2f}°, Prag permis: {MAX_SKEW_DEGREES}°)")
                print("-" * 60)
                print(">>> ACȚIUNE: Ajustați pragurile în secțiunea de configurare și rulați din nou. <<<")
                exit()
        
        print(f"Pre-filtrare: {len(valid_pages_indices)} din {len(final_results)} pagini considerate valide pentru analiză.\n")
    else:
        valid_pages_indices = list(range(len(final_results)))

    if not valid_pages_indices:
        print("Nicio pagină nu a trecut de pre-filtrare. Statisticile nu pot fi calculate.")
        exit()

    metrics_for_stats = []
    for i in valid_pages_indices:
        res = final_results[i]
        x1, y1, x2, y2 = res['box']
        is_recto = ((i % 2 == 0) and START_PAGE_IS_RECTO) or ((i % 2 != 0) and not START_PAGE_IS_RECTO)
        if is_recto: inner, outer = x1, img_width - x2
        else: inner, outer = img_width - x2, x1
        metrics_for_stats.append((inner, outer, y1, img_height - y2, x2 - x1, y2 - y1, abs(res['skew'])))

    final_stats = {}
    iqr_mask = None
    try:
        stats_on_valid, iqr_mask = compute_statistics(metrics_for_stats)
        final_stats = stats_on_valid
        final_stats['total_pages_detected'] = len(final_results)
        final_stats['pages_prefiltered'] = len(valid_pages_indices)
        final_stats['pages_for_median'] = int(np.sum(iqr_mask))
    except ValueError as e: 
        print(f"Eroare la calcularea statisticilor: {e}")
        exit()
    
    median_box_for_viz = (
        final_stats['inner_margin'][1], 
        final_stats['top_margin'][1], 
        img_width - final_stats['outer_margin'][1], 
        img_height - final_stats['bottom_margin'][1]
    )

    final_inlier_mask = [False] * len(final_results)
    for i, is_inlier_in_subset in enumerate(iqr_mask):
        if is_inlier_in_subset:
            original_index = valid_pages_indices[i]
            final_inlier_mask[original_index] = True

    print(f"Se pregătește vizualizarea suprapunerii '{output_overlay}'...")
    mirrored_boxes_for_overlay = []
    
    if VISUALIZATION_MODE == 'inliers':
        print(f"Mod 'inliers': se vor include doar cele {final_stats['pages_for_median']} pagini 'inlier'.")
        pages_to_include_mask = final_inlier_mask
    elif VISUALIZATION_MODE == 'outliers':
        num_outliers = len(valid_pages_indices) - final_stats['pages_for_median']
        print(f"Mod 'outliers': se vor include cele {num_outliers} pagini 'outlier' din setul pre-filtrat.")
        outlier_mask_from_valid = [not m for m in iqr_mask]
        pages_to_include_mask = [False] * len(final_results)
        for i, is_outlier in enumerate(outlier_mask_from_valid):
            if is_outlier:
                original_index = valid_pages_indices[i]
                pages_to_include_mask[original_index] = True
    else:
        print(f"Mod 'all': se vor include toate cele {final_stats['total_pages_detected']} pagini detectate.")
        pages_to_include_mask = [True] * len(final_results)

    for i, res in enumerate(final_results):
        if not pages_to_include_mask[i]: continue
        box = res['box']
        x1, y1, x2, y2 = box
        is_recto = ((i % 2 == 0) and START_PAGE_IS_RECTO) or ((i % 2 != 0) and not START_PAGE_IS_RECTO)
        if is_recto: mirrored_boxes_for_overlay.append(box)
        else: mirrored_boxes_for_overlay.append((img_width - x2, y1, img_width - x1, y2))
    
    box_to_draw = median_box_for_viz if VISUALIZATION_MODE in ['all', 'inliers'] else None
    visualize_text_blocks((img_height, img_width), mirrored_boxes_for_overlay, output_overlay, median_box=box_to_draw)
    print(f"Vizualizarea suprapunerii a fost salvată.")
    
    if DEBUG:
        print(f"Se generează imaginile de debug în directorul '{DEBUG_OUTPUT_DIR}'...")
        for i, res in enumerate(final_results):
            debug_image = cv2.imread(str(Path(DIRECTORY) / res['filename']))
            x1, y1, x2, y2 = map(int, res['box'])
            if i not in valid_pages_indices:
                color = (0, 0, 255)
            else:
                local_index = valid_pages_indices.index(i)
                color = (0, 255, 0) if iqr_mask[local_index] else (0, 165, 255)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 5) 
            cv2.putText(debug_image, f"Skew: {res['skew']:.2f} deg", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            cv2.imwrite(str(Path(DEBUG_OUTPUT_DIR) / res['filename']), debug_image)
        print("Imaginile de debug au fost generate cu succes.")

    print(f"\n--- Statistici Finale | Algoritm: {ALGORITHM} | Rezoluție: {DPI} DPI ---")
    page_w_mm, page_h_mm = pixels_to_mm(img_width, DPI), pixels_to_mm(img_height, DPI)
    print(f"{'Page_size':<22}: ({page_w_mm:.1f} x {page_h_mm:.1f}) mm")
    print("-" * 75)
    for key, value in final_stats.items():
        if key.startswith('pages_') or key.startswith('total_'): continue
        if key == 'skew': 
            min_val, med_val, max_val = value
            print(f"{key:<22}: ({min_val:5.2f}, {med_val:5.2f}, {max_val:5.2f}) deg")
        else: 
            min_mm, med_mm, max_mm = (pixels_to_mm(p, DPI) for p in value)
            print(f"{key:<22}: ({min_mm:5.1f}, {med_mm:5.1f}, {max_mm:5.1f}) mm")
    print("-" * 75)
    print(f"{'Pages total detected':<22}: {final_stats['total_pages_detected']}")
    if PREFILTER_ENABLED:
        print(f"{'Pages after pre-filter':<22}: {final_stats['pages_prefiltered']}")
    print(f"{'Pages used (inliers)':<22}: {final_stats['pages_for_median']}")
    print(f"{'Algorithm used':<22}: {ALGORITHM}")