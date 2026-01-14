# True Positives Visualization Scripts

This folder contains scripts to visualize all true positive loop closures on 3D trajectory paths.

## Scripts

### 1. `plot_all_true_positives.py`
Generates a single visualization showing all true positive loop closures for one model.

**Features:**
- Black trajectory path (prominent, thick line)
- Green loop closure connections (bright, visible lines)
- Red/green scatter points showing query and matched neighbors
- Statistics output (TP count, distances, segments)

**Usage:**
```bash
python plot_all_true_positives.py
```

**Configuration (edit in `main()`):**
- `dataset_root`: Path to dataset
- `predictions_path`: Path to predictions.pkl file
- `sequence`: Sequence name (e.g., "PCD_MED", "PCD_EASY")
- `model_name`: Model name (e.g., "PointNetPGAP")
- `topk`: Top-K predictions to consider (default: 1)
- `distance_threshold`: Max distance for loop closures in meters (default: 10.0)
- `min_temporal_distance`: Min frame gap for loop closures (default: 50)

**Output:**
- PNG file: `{sequence}_{model_name}_all_tp.png`
- Location: `/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives/`

---

### 2. `plot_tp_comparison.py`
Generates a multi-panel comparison visualization for multiple models.

**Features:**
- Side-by-side comparison of 5 models
- Black trajectory paths
- Green loop closure connections
- Each subplot shows model name and TP count
- Automatic layout (up to 5 columns)

**Usage:**
```bash
python plot_tp_comparison.py
```

**Configuration (edit in `main()`):**
- `dataset_root`: Path to dataset
- `saved_root`: Root directory for saved predictions
- `sequence`: Sequence to visualize
- `models`: List of model names to compare

**Models supported:**
- PointNetPGAP
- PointNetVLAD
- SPVSoAP3D (auto-handles naming: SPVSoAP3D-SoAP-log-pnl-fc-None)
- LOGG3D
- overlap_transformer

**Output:**
- PNG file: `{sequence}_all_models_tp_comparison.png`
- Location: `/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives/`

---

## Visualization Style

Both scripts follow the same visual style:

### Colors:
- **Black**: Trajectory path (thick, solid line)
- **Green**: Loop closure connections (prominent lines linking query to matched neighbor)
- **Red points**: Query frames with true positives
- **Green points**: Matched neighbor frames

### Parameters:
- `connection_alpha=0.8`: High visibility for green lines
- `connection_linewidth=2.0` (single) / `1.5` (comparison): Thick green lines
- Path linewidth: `2.5` (single) / `2.0` (comparison): Prominent black path

---

## File Structure Requirements

Predictions must be organized as:
```
saved/hortov2/{sequence}/{model}-None/predictions/place/{recall@1}/predictions.pkl
```

Example:
```
saved/hortov2/PCD_MED/PointNetPGAP-None/predictions/place/0.533@1/predictions.pkl
```

---

## Statistics Output

Both scripts print detailed statistics:
- Total true positive count
- Distance statistics (min, max, mean, median)
- Unique query frames with TPs
- True positives per segment

---

## Customization

### Adjust View Angle:
```python
view_angle=(elevation, azimuth)  # Default: (30, 45)
```

### Adjust Line Appearance:
```python
connection_alpha=0.8        # Transparency (0-1)
connection_linewidth=2.0    # Thickness
```

### Change Figure Size:
```python
figsize=(width, height)     # Default: (20, 16) single, (24, 6) comparison
```

---

## Example Output

**Single Model:**
- Clear visualization of all loop closures
- Easy to see density of matches
- Statistics for analysis

**Multi-Model Comparison:**
- Visual comparison of model performance
- See which models detect more TPs
- Identify patterns across models

---

## Notes

1. **Temporal Distance Filter**: Only considers loop closures with frame gap ≥ 50 frames to avoid trivial matches
2. **Distance Threshold**: Only includes predictions within specified distance (default 10m)
3. **Segment-Based TP**: True positive = predicted segment matches query segment
4. **Auto-Legend**: Legend only shows on first connection to avoid clutter

---

## Dependencies

- matplotlib
- numpy
- pickle
- PointNetGAP.dataloader.hortov2 (dataset, utils)

---

## Author Notes

These scripts generate publication-quality visualizations similar to ground truth loop closure figures, but using model predictions. The green-on-black style makes loop closures highly visible and easy to interpret.
