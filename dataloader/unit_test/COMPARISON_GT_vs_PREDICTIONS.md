# Comparison: Ground Truth vs Predictions Visualization

## Quick Reference

| Script | Data Source | Use Case | Key Parameters |
|--------|-------------|----------|----------------|
| `plot_loop_closure_gif.py` | Ground Truth | Show ideal loop closures | `distance_threshold`, `warm_up`, `lower_bound_idx` |
| `plot_loop_closure_pr_gif.py` | Model Predictions | Evaluate model performance | `topk`, `distance_threshold`, `model_names` |

## Visual Differences

### Ground Truth GIF
```
Title: "PCD_Easy_DARK - Loop Closure Detection"
Red star: Query frame
Green circles: True loop closures (spatially close frames)
Info: Frame ID, Label, Total neighbors found
```

### Predictions GIF
```
Title: "PCD_Easy_DARK_PointNetPGAP-None - Loop Closure Detection (PREDICTIONS)"
Red star: Query frame
Green circles: Predicted loop closures (top-K by descriptor similarity)
Info: Frame ID, Label, Top-K, Distance threshold, Predicted distances
```

## Code Comparison

### Ground Truth Approach

```python
# 1. Compute ground truth from trajectory
gt_lc = fs.get_ground_truth_loop_closure(
    warm_up=100,
    lower_bound_idx=50,
    distance_threshold=10.0,
    topk=1
)

# 2. Visualize ground truth
visualizer = LoopClosureVisualizer(fs, gt_lc, seq_name)
```

**Data Structure:**
```python
gt_lc = {
    'query_indices': array([...]),      # Query frame indices
    'neighbor_indices': array([...]),   # Neighbor frame indices
    'distances': array([...]),          # Spatial distances (meters)
    'valid_loop_closures': int          # Total count
}
```

### Predictions Approach

```python
# 1. Load model predictions
predictions = load_predictions('predictions.pkl')

# 2. Visualize predictions
visualizer = LoopClosureVisualizerPredictions(
    fs, predictions, seq_name,
    topk=1,  # Top-K predictions
    distance_threshold=10.0
)
```

**Data Structure:**
```python
predictions = {
    query_idx: {
        'pred_loops': {
            'idx': array([...]),        # Predicted neighbor indices
            'dist': array([...]),       # Descriptor distances
            'segment': array([...])     # Segment labels
        },
        'segment': int,                 # Query segment
        'true_loops': {...}             # Ground truth (for comparison)
    }
}
```

## When to Use Each

### Use Ground Truth Visualization When:
- ✓ Analyzing dataset characteristics
- ✓ Understanding spatial loop closure distribution
- ✓ Establishing baseline expectations
- ✓ Debugging trajectory data
- ✓ Creating dataset documentation

### Use Predictions Visualization When:
- ✓ Evaluating model performance
- ✓ Comparing different models
- ✓ Debugging model predictions
- ✓ Creating qualitative results for papers
- ✓ Analyzing top-K predictions
- ✓ Studying failure cases

## Example Use Cases

### Research Paper Figure
```python
# Show model predictions with ground truth overlay
# Use top-K=1 for cleaner visualization
TOPK = 1
DISTANCE_THRESHOLD = 10.0
SAMPLE_RATE = 20  # Sparse for clarity
```

### Model Debugging
```python
# Show multiple predictions to analyze failure modes
TOPK = 5
DISTANCE_THRESHOLD = 15.0
SAMPLE_RATE = 5  # Dense for detailed analysis
```

### Model Comparison
```python
# Generate GIFs for all models with same parameters
MODEL_NAMES = ["PointNetPGAP-None", "PointNetVLAD-None", "LOGG3D-None"]
TOPK = 1
DISTANCE_THRESHOLD = 10.0
```

## Parameter Guidelines

### Top-K Selection

| Top-K | Use Case | Visualization Clarity |
|-------|----------|----------------------|
| 1 | Clean visualization, single best match | ⭐⭐⭐⭐⭐ Excellent |
| 3 | Show alternative predictions | ⭐⭐⭐⭐ Good |
| 5 | Analyze prediction diversity | ⭐⭐⭐ Fair |
| 10+ | Deep debugging | ⭐⭐ Cluttered |

### Distance Threshold

| Threshold | Dataset Type | Purpose |
|-----------|--------------|---------|
| 5m | Dense trajectories | Strict loop closure |
| 10m | Normal trajectories | Standard evaluation |
| 15m | Sparse trajectories | Relaxed matching |
| 20m+ | Very sparse data | Maximum coverage |

### Sample Rate

| Sample Rate | Frames in GIF | GIF Duration (2 fps) | Use Case |
|-------------|---------------|----------------------|----------|
| 5 | ~80-100 | ~40-50s | Detailed analysis |
| 10 | ~40-50 | ~20-25s | Standard use |
| 20 | ~20-25 | ~10-12s | Quick overview |
| 50 | ~8-10 | ~4-5s | Fast preview |

## Output Comparison

### Ground Truth Outputs
```
dataset/PlaceRecognitionTestPolyTunnel/
  ├── loop_closure_PCD_EASY.gif
  ├── loop_closure_PCD_Easy_DARK.gif
  └── ...
```

### Predictions Outputs
```
saved/hortov2/loop_closure_gifs_predictions/
  ├── loop_closure_pred_PCD_EASY_PointNetPGAP-None_topk1.gif
  ├── loop_closure_pred_PCD_EASY_PointNetVLAD-None_topk1.gif
  ├── loop_closure_pred_PCD_EASY_LOGG3D-None_topk1.gif
  └── ...
```

## Performance Considerations

| Aspect | Ground Truth | Predictions |
|--------|--------------|-------------|
| Computation time | **Slower** (computes distances) | **Faster** (loads pre-computed) |
| Memory usage | **Higher** (distance matrix) | **Lower** (sparse predictions) |
| Disk space | Minimal (just GIF) | Requires predictions.pkl |
| Repeatability | Always same | Depends on model version |

## Workflow Example

### Complete Evaluation Workflow

```bash
# 1. Generate predictions (once per model)
python script_eval_hortov2.py

# 2. Generate ground truth GIFs (once per dataset)
python PointNetGAP/dataloader/unit_test/plot_loop_closure_gif.py

# 3. Generate prediction GIFs (once per model)
python PointNetGAP/dataloader/unit_test/plot_loop_closure_pr_gif.py

# 4. Compare side-by-side
# View ground truth: dataset/PlaceRecognitionTestPolyTunnel/loop_closure_*.gif
# View predictions: saved/hortov2/loop_closure_gifs_predictions/loop_closure_pred_*.gif
```

## Tips for Publication

### For Papers/Presentations:
1. Use **ground truth** to show dataset characteristics
2. Use **predictions** with top-K=1 for main results
3. Use **predictions** with top-K=5 to show failure analysis
4. Keep sample_rate=20+ for cleaner animations
5. Use 3-5 fps for presentations (slower = more readable)

### For Debugging:
1. Use **predictions** with multiple top-K values
2. Lower sample_rate (5-10) for detailed frame-by-frame analysis
3. Adjust distance threshold to investigate borderline cases
4. Compare against ground truth to identify systematic errors

## Common Questions

**Q: Why do prediction distances differ from spatial distances?**
A: Prediction 'dist' is descriptor similarity (smaller = more similar), not spatial distance. The script filters predictions by spatial distance after retrieval.

**Q: Can I show both predictions and ground truth together?**
A: Not in current version. Run both scripts separately and compare GIFs side-by-side.

**Q: What if my model predictions are empty?**
A: Check that evaluation ran successfully and saved predictions. The script will show "WARNING: No predictions available!"

**Q: How to speed up GIF generation?**
A: Increase `sample_rate` and reduce `dpi`. For quick previews, use sample_rate=50 and dpi=60.

**Q: Can I use this for other datasets?**
A: Yes, as long as the dataset follows the `file_structure` interface and predictions are in the expected format.

## Summary

- **Ground Truth** = Ideal loop closures based on spatial proximity
- **Predictions** = Model's actual loop closure detections
- Use **both** for comprehensive evaluation and debugging
- Adjust parameters based on visualization purpose (publication vs debugging)
