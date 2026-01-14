"""
Generate a combined grid figure with all sequences and models.

Creates a single figure with:
- Rows: Different sequences
- Columns: Ground Truth + Models
- Layout similar to academic paper figures
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageChops

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def crop_image_tight(img, manual_bbox=None, extra_crop_pixels=0):
    """
    Crop image tightly around non-transparent/non-white content.
    
    Args:
        img: PIL Image
        manual_bbox: Optional tuple (left, top, right, bottom) to use instead of auto-detection
                     If provided, will crop all images to this exact bounding box
        extra_crop_pixels: Additional pixels to crop from each edge after initial crop
                          Useful to remove any remaining borders/whitespace
    
    Returns:
        Cropped PIL Image
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Use manual bounding box if provided, otherwise auto-detect
    if manual_bbox is not None:
        bbox = manual_bbox
        print(f"Bounding box (manual): {bbox}")
    else:
        # Get the bounding box of non-transparent content
        # Create a mask of non-transparent pixels
        alpha = img.split()[-1]  # Get alpha channel
        bbox = alpha.getbbox()
        print(f"Bounding box (auto): {bbox}")
    
    if bbox:
        # Apply extra cropping if specified
        if extra_crop_pixels > 0:
            left, top, right, bottom = bbox
            bbox = (left + extra_crop_pixels, 
                   top + extra_crop_pixels,
                   right - extra_crop_pixels, 
                   bottom - extra_crop_pixels)
            print(f"  → After extra crop ({extra_crop_pixels}px): {bbox}")
        
        # Crop to content
        img_cropped = img.crop(bbox)
        return img_cropped
    
    return img


def load_image(image_path, manual_bbox=None, extra_crop_pixels=0):
    """
    Load an image from file and crop it tightly.
    
    Args:
        image_path: Path to the image file
        manual_bbox: Optional tuple (left, top, right, bottom) for manual cropping
        extra_crop_pixels: Additional pixels to crop from edges
    
    Returns:
        Cropped PIL Image or None if file not found
    """
    if os.path.exists(image_path):
        img = Image.open(image_path)
        # Crop tightly around content
        img_cropped = crop_image_tight(img, manual_bbox=manual_bbox, 
                                      extra_crop_pixels=extra_crop_pixels)
        return img_cropped
    return None


def create_combined_grid(input_dir, sequences, models, output_path, figsize=(20, 12), dpi=300, 
                        sequence_name_map=None, manual_bbox=None, extra_crop_pixels=0):
    """
    Create a combined grid figure with all sequences and models.
    
    Args:
        input_dir: Directory containing individual plot images
        sequences: List of sequence names
        models: List of model names (Ground Truth will be added automatically)
        output_path: Path to save the combined figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for output image
        sequence_name_map: Optional dict to map sequence names to display names
                          e.g., {"PCD_Easy_DARK": "OJ23", "PCD_MED": "ON22"}
        manual_bbox: Optional tuple (left, top, right, bottom) to crop all images uniformly
                     e.g., (700, 850, 2150, 1750) - useful for consistent cropping
        extra_crop_pixels: Additional pixels to crop from each edge (for removing borders)
                          e.g., 10 to crop 10px from all sides
    """
    # Add Ground Truth as first column
    all_columns = ["Ground Truth"] + models
    n_rows = len(sequences)
    n_cols = len(all_columns)
    
    print("=" * 80)
    print("Creating Combined Grid Figure")
    print("=" * 80)
    print(f"Sequences: {sequences}")
    print(f"Models: {all_columns}")
    print(f"Grid size: {n_rows} rows × {n_cols} columns")
    print(f"Crop mode: {'Manual' if manual_bbox else 'Auto-detect'}")
    if manual_bbox:
        print(f"Manual bbox: {manual_bbox}")
    if extra_crop_pixels > 0:
        print(f"Extra cropping: {extra_crop_pixels} pixels from each edge")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, 
                  left=0.04, right=0.995, top=0.98, bottom=0.02,
                  wspace=0.001, hspace=0.0)  # Zero vertical spacing between rows
    
    # Track missing images
    missing_images = []
    
    # Plot each cell in the grid
    for row_idx, sequence in enumerate(sequences):
        for col_idx, model in enumerate(all_columns):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Construct filename
            if model == "Ground Truth":
                filename = f"{sequence}_GroundTruth_loops.png"
            else:
                filename = f"{sequence}_{model}_tp.png"
            
            image_path = os.path.join(input_dir, filename)
            
            # Load and display image
            img = load_image(image_path, manual_bbox=manual_bbox, 
                           extra_crop_pixels=extra_crop_pixels)
            if img is not None:
                ax.imshow(img)
                print(f"  [{row_idx+1},{col_idx+1}] Loaded & cropped: {filename}")
            else:
                # If image not found, show empty plot with message
                ax.text(0.5, 0.5, 'Image\nNot Found', 
                       ha='center', va='center', fontsize=10, color='red')
                missing_images.append(filename)
                print(f"  [{row_idx+1},{col_idx+1}] MISSING: {filename}")
            
            # Remove axes ticks and spines for seamless appearance
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove borders for tighter layout
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add column labels (model names) on top row
            if row_idx == 0:
                # Format model name for display
                if model == "Ground Truth":
                    display_name = "Ground Truth"
                elif model == "PointNetPGAP":
                    display_name = "PointNetPGAP"
                elif model == "PointNetVLAD":
                    display_name = "PointNetVLAD"
                elif model == "overlap_transformer":
                    display_name = "OverlapTransformer"
                else:
                    display_name = model
                
                ax.set_title(display_name, fontsize=11, fontweight='bold', pad=5)
            
            # Add row labels (sequence names) on left column
            if col_idx == 0:
                # Use custom mapping if provided, otherwise use defaults
                if sequence_name_map and sequence in sequence_name_map:
                    display_name = sequence_name_map[sequence]
                else:
                    # Default mapping - User should verify these match their dataset
                    if sequence == "PCD_EASY":
                        display_name = "PCD_EASY"  # Easy sequence from Oct 2022
                    elif sequence == "PCD_Easy_DARK":
                        display_name = "PCD_Easy_DARK"  # Easy Dark sequence from Oct 2023
                    elif sequence == "PCD_MED":
                        display_name = "PCD_MED"  # Medium sequence from Oct/Nov 2022
                    elif sequence == "PCD_RAS_EASY":
                        display_name = "PCD_RAS_EASY"  # RAS Easy sequence from Sept/July 2023
                    else:
                        display_name = sequence
                
                ax.set_ylabel(display_name, fontsize=11, fontweight='bold', rotation=0, 
                            labelpad=35, ha='right', va='center')
    
    # Add main title
    # fig.suptitle('True Positive Predictions of Top-1 Retrieved Candidates', 
    #             fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    print("\n" + "=" * 80)
    print("Saving combined figure...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.0)
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.0)
    
    plt.close(fig)
    
    print(f"✓ Saved PNG: {output_path}")
    print(f"✓ Saved PDF: {pdf_path}")
    
    # Report missing images
    if missing_images:
        print("\n" + "!" * 80)
        print(f"WARNING: {len(missing_images)} images were not found:")
        for img in missing_images:
            print(f"  - {img}")
        print("!" * 80)
    else:
        print("\n✓ All images loaded successfully!")
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


def main():
    """Main function."""
    
    # Configuration
    input_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives_individual"
    output_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots"
    output_filename = "combined_true_positives_grid.png"
    
    # Sequences (in order of rows)
    sequences = [
        "PCD_Easy_DARK",  # Will be displayed as OJ23
        "PCD_MED",        # Will be displayed as ON22
        # "PCD_EASY",     # OJ22 - Uncomment if available
        # "PCD_RAS_EASY"  # SJ23 - Uncomment if available
    ]
    
    # OPTIONAL: Custom sequence name mapping
    # Override default mappings by uncommenting and modifying:
    sequence_name_map = {
        "PCD_Easy_DARK": "OJ23",
        "PCD_MED": "ON22",
        # "PCD_EASY": "OJ22",
        # "PCD_RAS_EASY": "SJ23"
    }
    
    # OPTIONAL: Manual bounding box for uniform cropping
    # Format: (left, top, right, bottom) in pixels
    # Example: (700, 850, 2150, 1750)
    # Set to None for automatic detection per image
    manual_bbox = (692, 838, 2178, 1767)
    # To use manual cropping, uncomment and adjust:
    # manual_bbox = (690, 838, 2180, 1770)  # Adjust based on your images
    
    # OPTIONAL: Extra cropping to remove borders/whitespace
    # This crops additional pixels from all edges after the main crop
    # Useful if there's still visible space between rows
    extra_crop_pixels = 20  # Try 10-30 pixels
    # Set to 0 for no extra cropping:
    # extra_crop_pixels = 0
    
    # Models (in order of columns, after Ground Truth)
    models = [
        "PointNetPGAP",
        "PointNetVLAD",
        "SPVSoAP3D",
        "overlap_transformer",
        "LOGG3D"
    ]
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path
    output_path = os.path.join(output_dir, output_filename)
    
    # Create combined grid
    create_combined_grid(
        input_dir=input_dir,
        sequences=sequences,
        models=models,
        output_path=output_path,
        figsize=(24, 8),  # Width, Height in inches
        dpi=300,
        sequence_name_map=sequence_name_map,
        manual_bbox=manual_bbox,
        extra_crop_pixels=extra_crop_pixels
    )


if __name__ == "__main__":
    main()
