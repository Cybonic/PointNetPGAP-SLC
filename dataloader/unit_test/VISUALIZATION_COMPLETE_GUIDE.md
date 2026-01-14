# True Positive Loop Closure Visualization - Complete Guide

This folder contains scripts for visualizing true positive loop closures from place recognition model predictions.

---

## 📁 Available Scripts

### 1. **plot_tp_individual.py** ⭐ RECOMMENDED
**Purpose**: Generate individual plots for each model-sequence combination  
**Style**: Black trajectory + Green loop closures + No grid  
**Output**: One PNG per model-sequence  

**Use when:**
- You need separate plots for each model
- You want consistent viewpoints across all plots
- You're preparing figures for publication
- You need clean, grid-free visualizations

---

### 2. **plot_all_true_positives.py**
**Purpose**: Generate a single plot for one model-sequence  
**Style**: Black trajectory + Green loop closures  
**Output**: One PNG with statistics  

**Use when:**
- You want to analyze one specific model
- You need detailed statistics output
- You're exploring a single case

---

### 3. **plot_tp_comparison.py**
**Purpose**: Multi-panel comparison of models  
**Style**: Side-by-side subplots  
**Output**: One PNG with all models  

**Use when:**
- You want to compare models visually in one figure
- You need a quick overview of all models
- You're creating a comparison figure for papers

---

### 4. **example_viewpoint_comparison.py**
**Purpose**: Test different camera angles  
**Style**: Same data, multiple viewpoints  
**Output**: Multiple sets of plots  

**Use when:**
- You're unsure what viewpoint works best
- You want to explore different angles
- You're finding the optimal visualization

---

## 🎯 Quick Start

### For Publication (Recommended)
```bash
cd /home/tiago/workspace/place_uk/PointNetGAP
python dataloader/unit_test/plot_tp_individual.py
```

Edit the configuration in `main()`:
```python
# Set consistent viewpoint
view_elev = 30    # Elevation angle
view_azim = 45    # Azimuth angle

# Clean style
show_grid = False
show_legend = True
show_axes = True
```

**Result**: Individual plots for each model-sequence with consistent styling

---

### For Quick Exploration
```bash
python dataloader/unit_test/plot_all_true_positives.py
```

**Result**: Single plot with detailed statistics

---

### For Multi-Model Overview
```bash
python dataloader/unit_test/plot_tp_comparison.py
```

**Result**: One figure with all models side-by-side

---

### For Finding Best Viewpoint
```bash
python dataloader/unit_test/example_viewpoint_comparison.py
```

**Result**: Same plots rendered from 6 different angles

---

## 🎨 Visualization Style

### Common Style Across All Scripts

**Colors:**
- **Black (`'k'`)**: Trajectory path (prominent, thick)
- **Green (`'g'`)**: Loop closure connections (bright, visible)
- **White**: Background

**Lines:**
- Path: `linewidth=2.5`, `alpha=1.0`
- Connections: `linewidth=2.0`, `alpha=0.8`

**Grid:** Disabled by default for clean look

---

## ⚙️ Key Parameters

### Viewpoint (MOST IMPORTANT for consistency)
```python
view_elev = 30    # Vertical angle (0-90)
view_azim = 45    # Horizontal rotation (0-360)
```

**Common combinations:**
- `(30, 45)` - Standard isometric
- `(90, 0)` - Top-down view
- `(60, 45)` - High angle
- `(30, 60)` - More diagonal

### Loop Closure Filtering
```python
topk = 1                      # Top-K predictions
distance_threshold = 10.0     # Max distance (meters)
min_temporal_distance = 50    # Min frame gap
```

### Display Options
```python
show_grid = False     # Grid on/off
show_legend = True    # Legend on/off
show_axes = True      # Axis labels on/off
```

---

## 📊 Output Examples

### plot_tp_individual.py
```
plots/true_positives_individual/
├── PCD_MED_PointNetPGAP_tp.png        (523 TPs)
├── PCD_MED_PointNetVLAD_tp.png        (461 TPs)
├── PCD_MED_SPVSoAP3D_tp.png           (337 TPs)
├── PCD_MED_LOGG3D_tp.png              (185 TPs)
├── PCD_MED_overlap_transformer_tp.png (152 TPs)
├── PCD_Easy_DARK_PointNetPGAP_tp.png  (367 TPs)
└── ...
```

### plot_tp_comparison.py
```
plots/true_positives/
└── PCD_MED_all_models_tp_comparison.png  (5 subplots)
```

---

## 🔧 Configuration Guide

### For Consistent Publication Figures

**Step 1**: Choose viewpoint by testing
```bash
python example_viewpoint_comparison.py
```

**Step 2**: Set viewpoint in plot_tp_individual.py
```python
view_elev = 30  # Your chosen elevation
view_azim = 45  # Your chosen azimuth
```

**Step 3**: Generate all plots
```bash
python plot_tp_individual.py
```

**Step 4**: All plots will have identical viewpoints ✓

---

### For Different Sequences, Same View

Keep `view_elev` and `view_azim` constant:
```python
sequences = ["PCD_Easy_DARK", "PCD_MED", "PCD_EASY"]
view_elev = 30  # Same for all
view_azim = 45  # Same for all
```

All plots can now be directly compared!

---

## 📈 Understanding True Positives

### What is a True Positive?
A loop closure prediction where:
1. Predicted neighbor is within `distance_threshold` meters
2. Frame gap is ≥ `min_temporal_distance` frames
3. **Predicted segment matches query segment** ✓

### Why These Filters?
- **Distance threshold**: Removes false matches that are too far
- **Temporal distance**: Removes trivial matches (consecutive frames)
- **Segment matching**: Ensures prediction is semantically correct

---

## 🚀 Performance Tips

### Fast Generation
```python
sequences = ["PCD_MED"]           # Test with one sequence first
models = ["PointNetPGAP"]         # Test with one model first
```

### Full Production
```python
sequences = ["PCD_Easy_DARK", "PCD_MED"]
models = ["PointNetPGAP", "PointNetVLAD", "SPVSoAP3D", "LOGG3D", "overlap_transformer"]
```

**Time estimates:**
- 1 model × 1 sequence: ~2 seconds
- 5 models × 2 sequences: ~20 seconds
- 5 models × 4 sequences: ~40 seconds

---

## 📝 File Organization

### Input Requirements
```
saved/hortov2/
└── {sequence}/
    └── {model}-None/
        └── predictions/
            └── place/
                └── {X.XXX@1}/
                    └── predictions.pkl
```

### Output Structure
```
plots/true_positives_individual/
├── {sequence}_{model}_tp.png
└── ...
```

---

## 🎓 Best Practices

### ✅ DO:
- Use consistent viewpoints across all plots
- Disable grid for publication figures
- Test viewpoints with example script first
- Use high DPI (300-600) for print

### ❌ DON'T:
- Change viewpoint between sequences (unless intentional)
- Use grid for publication (looks cluttered)
- Use very small figure sizes (hard to see details)
- Mix different `topk` or `distance_threshold` values

---

## 🔍 Troubleshooting

### No predictions found?
Check directory structure matches expected format.

### Different camera angles?
Verify `view_elev` and `view_azim` are identical in all runs.

### Grid still showing?
Set `show_grid=False` in configuration.

### Plots look different?
Ensure `figsize` and viewpoint are consistent.

---

## 📚 Documentation Files

- **INDIVIDUAL_PLOTS_GUIDE.md** - Detailed guide for plot_tp_individual.py
- **TRUE_POSITIVES_VISUALIZATION_README.md** - Overview of all visualization scripts
- This file - Complete guide to all scripts

---

## 🎯 Recommended Workflow

1. **Explore**: Use `plot_all_true_positives.py` to understand one model
2. **Compare**: Use `plot_tp_comparison.py` for quick multi-model overview
3. **Test Views**: Use `example_viewpoint_comparison.py` to find best angle
4. **Generate**: Use `plot_tp_individual.py` for final publication figures

---

## 💡 Pro Tips

### Tip 1: Consistent Camera Angles
Save your preferred viewpoint in a config file:
```python
# config.py
VIEWPOINT = {"elev": 30, "azim": 45}
```

### Tip 2: Batch Processing
Process all sequences at once for consistency.

### Tip 3: High-Res for Papers
```python
# Modify savefig in the script:
plt.savefig(output_path, dpi=600, bbox_inches='tight')
```

### Tip 4: Minimal Style
For super clean figures:
```python
show_grid = False
show_legend = False
show_axes = False
```

---

## 📧 Questions?

Check the individual script documentation files for more details.

---

**Last Updated**: January 14, 2026  
**Scripts Version**: 1.0
