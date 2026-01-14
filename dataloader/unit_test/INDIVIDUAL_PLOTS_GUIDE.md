# Individual True Positive Plots Generator

**Script:** `plot_tp_individual.py`

This script generates individual publication-quality plots for each model-sequence combination, showing all true positive loop closures with consistent viewpoints and clean styling.

---

## Features

✅ **Individual Plots**: One plot per model-sequence combination  
✅ **Consistent Viewpoint**: All plots use the same camera angle  
✅ **No Grid**: Clean background without grid lines  
✅ **Black Trajectory**: Prominent path in black  
✅ **Green Loop Closures**: Bright connection lines  
✅ **Configurable Parameters**: Easy to customize all settings  
✅ **Batch Processing**: Automatically processes all combinations  
✅ **Summary Report**: Shows success/failure for each plot  

---

## Usage

```bash
cd /home/tiago/workspace/place_uk/PointNetGAP
python dataloader/unit_test/plot_tp_individual.py
```

---

## Configuration

All parameters are in the `main()` function's **CONFIGURATION SECTION**:

### Paths
```python
dataset_root = "/path/to/PlaceRecognitionTestPolyTunnel"
saved_root = "/path/to/saved/hortov2"
output_dir = "/path/to/output/plots"
```

### Sequences and Models
```python
sequences = [
    "PCD_Easy_DARK",
    "PCD_MED"
]

models = [
    "PointNetPGAP",
    "PointNetVLAD",
    "SPVSoAP3D",
    "LOGG3D",
    "overlap_transformer"
]
```

### Loop Closure Parameters
```python
topk = 1                      # Top-K predictions to consider
distance_threshold = 10.0     # Max distance in meters
min_temporal_distance = 50    # Min frame gap
```

### Viewpoint Parameters (CRITICAL FOR CONSISTENCY)
```python
view_elev = 30               # Elevation angle (degrees)
view_azim = 45               # Azimuth angle (degrees)
```

**Common viewpoint angles:**
- **Standard**: `elev=30, azim=45` - Good overall view
- **Top-down**: `elev=90, azim=0` - Bird's eye view
- **Side view**: `elev=0, azim=0` - Horizontal perspective
- **Isometric**: `elev=30, azim=60` - Technical drawing style

### Display Options
```python
figsize = (16, 12)           # Figure size (width, height)
show_grid = False            # Show/hide background grid
show_legend = True           # Show/hide legend
show_axes = True             # Show/hide axis labels
```

### Line Appearance
```python
connection_alpha = 0.8       # Transparency (0=invisible, 1=solid)
connection_linewidth = 2.0   # Thickness of green lines
```

---

## Output

### File Naming
```
{sequence}_{model}_tp.png
```

Examples:
- `PCD_MED_PointNetPGAP_tp.png`
- `PCD_Easy_DARK_PointNetVLAD_tp.png`
- `PCD_MED_SPVSoAP3D_tp.png`

### Output Directory Structure
```
plots/true_positives_individual/
├── PCD_Easy_DARK_PointNetPGAP_tp.png
├── PCD_Easy_DARK_PointNetVLAD_tp.png
├── PCD_Easy_DARK_SPVSoAP3D_tp.png
├── PCD_Easy_DARK_LOGG3D_tp.png
├── PCD_Easy_DARK_overlap_transformer_tp.png
├── PCD_MED_PointNetPGAP_tp.png
├── PCD_MED_PointNetVLAD_tp.png
├── PCD_MED_SPVSoAP3D_tp.png
├── PCD_MED_LOGG3D_tp.png
└── PCD_MED_overlap_transformer_tp.png
```

---

## Visual Style

### Elements
- **Black path**: Solid line (linewidth=2.5)
- **Green connections**: Loop closures (linewidth=2.0, alpha=0.8)
- **White background**: Clean, publication-ready
- **No grid**: Removed for cleaner look
- **Title**: Model name, sequence, and TP count

### Color Scheme
- `'k'` (black): Trajectory path
- `'g'` (green): Loop closure connections
- White: Background and panes

---

## Example Output Summary

```
================================================================================
SUMMARY
================================================================================
Sequence             Model                     Status               TPs       
--------------------------------------------------------------------------------
PCD_Easy_DARK        PointNetPGAP              SUCCESS              367       
PCD_Easy_DARK        PointNetVLAD              SUCCESS              347       
PCD_Easy_DARK        SPVSoAP3D                 SUCCESS              299       
PCD_Easy_DARK        LOGG3D                    SUCCESS              234       
PCD_Easy_DARK        overlap_transformer       SUCCESS              217       
PCD_MED              PointNetPGAP              SUCCESS              523       
PCD_MED              PointNetVLAD              SUCCESS              461       
PCD_MED              SPVSoAP3D                 SUCCESS              337       
PCD_MED              LOGG3D                    SUCCESS              185       
PCD_MED              overlap_transformer       SUCCESS              152       
--------------------------------------------------------------------------------
Total successful plots: 10/10
```

---

## Troubleshooting

### Issue: "Model directory not found"
**Solution**: Check that predictions exist for that sequence/model combination.

### Issue: "No @1 directory found"
**Solution**: Predictions may not be organized correctly. Check the directory structure:
```
saved/hortov2/{sequence}/{model}-None/predictions/place/{X.XXX@1}/predictions.pkl
```

### Issue: "pose file does not exist"
**Solution**: The sequence path in the dataset is incorrect. Check dataset organization.

### Issue: All plots look different
**Solution**: Ensure `view_elev` and `view_azim` are the same for all runs.

---

## Advanced Usage

### Different Viewpoints for Different Sequences

If you need different viewpoints per sequence:

```python
viewpoint_config = {
    "PCD_EASY": (30, 45),
    "PCD_MED": (30, 60),
    "PCD_Easy_DARK": (45, 45)
}

# In generate_all_plots(), pass sequence-specific angles
view_elev, view_azim = viewpoint_config.get(sequence, (30, 45))
```

### Minimal Style (No Legend, No Axes)

For a super clean look:
```python
show_grid = False
show_legend = False
show_axes = False
```

### High-Resolution Output

For publications:
```python
# In plot_model_sequence(), modify the savefig call:
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
```

### Different Figure Sizes per Sequence

```python
figsize_config = {
    "PCD_EASY": (12, 10),
    "PCD_MED": (16, 12),
}

figsize = figsize_config.get(sequence, (16, 12))
```

---

## Recommended Viewpoints

### For Greenhouse/Tunnel Environments
```python
view_elev = 30    # Slightly elevated
view_azim = 45    # Diagonal view
```

### For Outdoor/Field Environments
```python
view_elev = 45    # Higher elevation
view_azim = 60    # More rotation
```

### For Long Straight Paths
```python
view_elev = 20    # Lower elevation
view_azim = 0     # Straight-on view
```

---

## Integration with Other Scripts

### Use with Comparison Script
1. Generate individual plots with `plot_tp_individual.py`
2. Use `plot_tp_comparison.py` for multi-panel comparison
3. Both will use consistent styling

### Use with Ground Truth
- Set `topk=1` and `distance_threshold=10.0` to match ground truth parameters
- Compare with ground truth visualizations

---

## Performance

- **Processing time**: ~2-3 seconds per plot
- **Memory usage**: ~500MB peak
- **Output size**: ~1-2MB per PNG (300 DPI)

For 10 plots: ~20-30 seconds total

---

## File Requirements

### Predictions File Structure
```
predictions.pkl = {
    query_idx: {
        'pred_loops': {
            'idx': np.array([...]),      # Neighbor indices
            'dist': np.array([...]),     # Distances
            'segment': np.array([...])   # Segment IDs
        },
        'segment': int,                  # Query segment
        'true_loops': {...}              # Ground truth (optional)
    },
    ...
}
```

### Dataset Requirements
- Must have position data (x, y, z)
- Must have label/segment information
- Must support `file_structure` class from hortov2 dataloader

---

## Tips for Publication-Quality Figures

1. **Consistent Viewpoint**: Use the same `view_elev` and `view_azim` for all plots
2. **No Grid**: Set `show_grid=False` for cleaner look
3. **High DPI**: Use `dpi=300` or `dpi=600` for print quality
4. **White Background**: Default `facecolor='white'` is best for papers
5. **Clear Titles**: Model name and TP count provide context
6. **Legend Position**: Upper right avoids data overlap

---

## Quick Start Example

Minimal configuration for quick testing:

```python
sequences = ["PCD_MED"]
models = ["PointNetPGAP", "PointNetVLAD"]
view_elev = 30
view_azim = 45
show_grid = False
```

This will generate 2 plots quickly for testing.

---

## Author Notes

This script is designed to generate consistent, publication-ready visualizations for comparing place recognition models across different sequences. The key feature is the unified viewpoint parameter that ensures all plots can be directly compared visually.
