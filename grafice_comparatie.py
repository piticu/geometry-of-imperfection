import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The input file containing the merged 'before' and 'after' data.
COMPARISON_FILE = "comparatie_scantailor.csv"

# The directory where the output graphs will be saved.
OUTPUT_DIRECTORY = "grafice_comparatie"
# ==============================================================================
# END CONFIGURATION
# ==============================================================================

def plot_skew_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a Kernel Density Estimate (KDE) plot showing the
    distribution of absolute skew before and after processing.
    """
    print("Generating Graph 1: Skew Distribution...")
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(df['abs_skew_before'], label='Before ScanTailor', fill=True, alpha=0.5, linewidth=2)
    sns.kdeplot(df['abs_skew_after'], label='After ScanTailor', fill=True, alpha=0.5, linewidth=2, color='green')
    
    plt.title('Distribution of Absolute Page Skew (Before vs. After ScanTailor)', fontsize=16)
    plt.xlabel('Absolute Skew (degrees)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = output_dir / "1_skew_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f" -> Saved to {output_path}")

def plot_dimension_consistency(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves box plots to compare the consistency of content
    width and height before and after processing.
    """
    print("Generating Graph 2: Content Dimension Consistency...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Prepare data for box plots
    width_data = df[['width_before', 'width_after']]
    width_data.columns = ['Before', 'After']
    
    height_data = df[['height_before', 'height_after']]
    height_data.columns = ['Before', 'After']
    
    # Width Box Plot
    sns.boxplot(data=width_data, ax=axes[0], palette="pastel")
    axes[0].set_title('Content Width Consistency', fontsize=14)
    axes[0].set_ylabel('Width (pixels)', fontsize=12)
    
    # Height Box Plot
    sns.boxplot(data=height_data, ax=axes[1], palette="pastel")
    axes[1].set_title('Content Height Consistency', fontsize=14)
    axes[1].set_ylabel('Height (pixels)', fontsize=12)
    
    fig.suptitle('Content Dimension Comparison (Before vs. After ScanTailor)', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    output_path = output_dir / "2_dimension_consistency.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f" -> Saved to {output_path}")

def plot_content_shift(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a scatter plot showing the vector shift of the
    content's top-left corner (x1, y1).
    """
    print("Generating Graph 3: Content Shift Vector Plot...")
    plt.figure(figsize=(10, 10))
    
    # We use delta_x1 and delta_y1
    plt.scatter(df['delta_x1'], df['delta_y1'], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add a reference point at the origin
    plt.scatter(0, 0, color='red', s=100, zorder=5, label='Origin (No Shift)')
    
    # Add crosshairs for reference
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    
    plt.title('Shift Vector of Content Block Top-Left Corner', fontsize=16)
    plt.xlabel('Horizontal Shift (delta_x1) in pixels\n(Right ->)', fontsize=12)
    plt.ylabel('Vertical Shift (delta_y1) in pixels\n(Down ->)', fontsize=12)
    
    # Invert y-axis because image coordinates have y increasing downwards
    # This makes the plot intuitive: a positive delta_y means the content moved DOWN.
    plt.gca().invert_yaxis()
    
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.axis('equal') # Ensure a 1px shift on x-axis looks the same as a 1px shift on y-axis
    plt.tight_layout()
    
    output_path = output_dir / "3_content_shift.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f" -> Saved to {output_path}")

def generate_graphs():
    """
    Main function to load data and generate all comparison graphs.
    """
    comparison_file = Path(COMPARISON_FILE)
    output_dir = Path(OUTPUT_DIRECTORY)

    if not comparison_file.exists():
        print(f"ERROR: Comparison file not found at '{comparison_file}'")
        print("Please run 'compara_rezultate.py' first to generate it.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from '{comparison_file}'...")
    df = pd.read_csv(comparison_file)

    # Generate all graphs
    plot_skew_distribution(df, output_dir)
    plot_dimension_consistency(df, output_dir)
    plot_content_shift(df, output_dir)

    print(f"\nAll graphs have been successfully generated in the '{output_dir}' folder.")

if __name__ == "__main__":
    generate_graphs()