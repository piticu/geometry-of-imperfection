import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The CSV file with results BEFORE ScanTailor processing
FILE_BEFORE = "rezultate_inainte_de_scantailor.csv"

# The CSV file with results AFTER ScanTailor processing
FILE_AFTER = "rezultate_dupa_scantailor.csv"

# The final output file containing the comparison
OUTPUT_COMPARISON_FILE = "comparatie_scantailor.csv"

# The columns we want to compare and calculate the difference for
COLUMNS_TO_COMPARE = [
    'x1', 'y1', 'x2', 'y2', 
    'width', 'height', 'skew', 'abs_skew'
]
# ==============================================================================
# END CONFIGURATION
# ==============================================================================

def compare_scan_results():
    """
    Loads 'before' and 'after' data, merges them, calculates the differences,
    and saves a comparison report.
    """
    try:
        # Load the two CSV files into pandas DataFrames
        df_before = pd.read_csv(FILE_BEFORE)
        df_after = pd.read_csv(FILE_AFTER)
        print(f"--- Comparison Analysis ---")
        print(f"Loaded {len(df_before)} records from '{FILE_BEFORE}' (Before ScanTailor)")
        print(f"Loaded {len(df_after)} records from '{FILE_AFTER}' (After ScanTailor)")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e}")
        print("Please make sure both CSV files exist in the same directory as the script.")
        return

    # Merge the two DataFrames based on the 'filename' column
    # We add suffixes to distinguish between 'before' and 'after' columns
    df_comparison = pd.merge(
        df_before, 
        df_after, 
        on='filename', 
        suffixes=('_before', '_after')
    )
    
    # Filter for only the columns we are interested in, for clarity
    # This step is optional but makes the final CSV cleaner
    columns_to_keep = ['filename']
    for col in COLUMNS_TO_COMPARE:
        columns_to_keep.append(f"{col}_before")
        columns_to_keep.append(f"{col}_after")
    
    # Ensure all columns exist before trying to select them
    existing_columns = [col for col in columns_to_keep if col in df_comparison.columns]
    df_comparison = df_comparison[existing_columns]

    print(f"\nMerged {len(df_comparison)} common records based on 'filename'.")
    
    # Calculate the 'delta' (difference) for each comparable column
    print("Calculating 'delta' columns (after - before)...")
    for col in COLUMNS_TO_COMPARE:
        col_before = f"{col}_before"
        col_after = f"{col}_after"
        
        # Check if both before and after columns exist in the merged dataframe
        if col_before in df_comparison.columns and col_after in df_comparison.columns:
            df_comparison[f'delta_{col}'] = df_comparison[col_after] - df_comparison[col_before]

    # --- Analysis & Reporting ---
    print("\n--- Key Changes Analysis ---")

    # 1. Analyze Skew
    avg_skew_before = df_comparison['abs_skew_before'].mean()
    avg_skew_after = df_comparison['abs_skew_after'].mean()
    print(f"\n1. Skew (Absolute Average):")
    print(f"   - Before: {avg_skew_before:.2f} degrees")
    print(f"   - After:  {avg_skew_after:.2f} degrees")
    if avg_skew_after < avg_skew_before:
        reduction = (avg_skew_before - avg_skew_after) / avg_skew_before * 100
        print(f"   => SUCCESS: Average skew was reduced by {reduction:.1f}%.")
    else:
        print(f"   => WARNING: Average skew has increased.")

    # 2. Analyze Size Consistency
    width_std_before = df_comparison['width_before'].std()
    width_std_after = df_comparison['width_after'].std()
    height_std_before = df_comparison['height_before'].std()
    height_std_after = df_comparison['height_after'].std()
    print(f"\n2. Content Size Consistency (Standard Deviation):")
    print(f"   - Width STD Before: {width_std_before:.1f} | After: {width_std_after:.1f}")
    if width_std_after < width_std_before:
        print(f"   => SUCCESS: Content width is now more consistent.")
    print(f"   - Height STD Before: {height_std_before:.1f} | After: {height_std_after:.1f}")
    if height_std_after < height_std_before:
        print(f"   => SUCCESS: Content height is now more consistent.")

    # 3. Show files with the biggest changes in skew
    df_comparison['abs_delta_skew'] = abs(df_comparison['delta_abs_skew'])
    top_5_skew_changes = df_comparison.sort_values(by='abs_delta_skew', ascending=False).head(5)
    print("\n3. Top 5 Files with Largest Change in Absolute Skew:")
    print(top_5_skew_changes[['filename', 'abs_skew_before', 'abs_skew_after', 'delta_abs_skew']].round(2))

    try:
        # Save the final comparison DataFrame to a CSV file
        df_comparison.to_csv(OUTPUT_COMPARISON_FILE, index=False, float_format='%.2f')
        print(f"\nSUCCESS: Full comparison report saved to '{OUTPUT_COMPARISON_FILE}'.")
        print("You can now open this file in Excel or another spreadsheet program for detailed analysis.")
    except Exception as e:
        print(f"\nERROR: Could not save the comparison file: {e}")

if __name__ == "__main__":
    compare_scan_results()