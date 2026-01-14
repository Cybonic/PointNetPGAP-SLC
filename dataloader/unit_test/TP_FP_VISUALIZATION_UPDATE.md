# TP/FP Visualization Update

## Changes Made

The loop closure visualization script has been updated to distinguish between **True Positives (TP)** and **False Positives (FP)** in predictions.

## Visual Changes

### Color Coding
- **Green circles with "TP" label**: True Positives (predicted segment matches query segment)
- **Red circles with "FP" label**: False Positives (predicted segment does NOT match query segment)
- **Green lines**: Connections to True Positives
- **Red lines**: Connections to False Positives

### Information Display

#### Title
Before: `Frame X | Top-K | Loop closures found: N`
After: `Frame X | Top-K | TP: X | FP: Y`

#### Left Info Box (Stats)
- Total frames
- Frame ID
- **Query Segment** (new)
- Top-K
- Distance threshold
- **True Positives count** (new)
- **False Positives count** (new)

#### Right Info Box (Predictions)
Before: Just distances
After: Distance + Segment + TP/FP label
```
Predictions:
  1. 0.0120m | Seg:0 | TP
  2. 0.0460m | Seg:0 | TP
  3. 0.0540m | Seg:1 | FP
  4. 0.0550m | Seg:1 | FP
  5. 0.0560m | Seg:1 | FP
```

#### Updated Legend
- Trajectory up to query (blue line)
- Query frame (red star)
- **True Positive (TP)** - green circle (new)
- **False Positive (FP)** - red circle (new)

## Classification Logic

### True Positive (TP)
A prediction is classified as TP when:
```python
predicted_segment == query_segment
```

### False Positive (FP)
A prediction is classified as FP when:
```python
predicted_segment != query_segment
```

## Technical Implementation

### Updated `get_frame_data()` method
Now returns:
```python
{
    'query_idx': int,
    'query_segment': int,              # NEW
    'neighbor_indices': list,
    'neighbor_distances': list,
    'neighbor_segments': list,         # NEW
    'is_true_positive': list,          # NEW (boolean array)
    'num_neighbors': int,
    'num_true_positives': int,         # NEW
    'num_false_positives': int         # NEW
}
```

### Updated `plot_frame()` method
- Calculates `max_range` early for label positioning
- Loops through neighbors with TP/FP classification
- Colors and labels each prediction appropriately
- Adds text labels ("TP" or "FP") above each predicted neighbor

## Usage

The script works exactly the same way as before:

```bash
python plot_loop_closure_pr_gif.py
```

Parameters remain the same:
- `TOPK`: Number of top predictions to show
- `DISTANCE_THRESHOLD`: Maximum spatial distance (meters)
- `SAMPLE_RATE`: Frame sampling rate

## Example Interpretation

### Scenario 1: All True Positives
```
Frame 495 | Top-3 | TP: 3 | FP: 0
```
- All 3 predicted neighbors are in the same segment as the query
- All circles are **green** with "TP" labels
- All connection lines are **green**
- Model is performing well for this query

### Scenario 2: Mixed Results
```
Frame 520 | Top-5 | TP: 2 | FP: 3
```
- 2 predictions are correct (same segment)
- 3 predictions are incorrect (different segments)
- You'll see **2 green circles** and **3 red circles**
- Indicates model has some confusion for this query

### Scenario 3: All False Positives
```
Frame 550 | Top-5 | TP: 0 | FP: 5
```
- All predictions are wrong segments
- All circles are **red** with "FP" labels
- All connection lines are **red**
- Model completely failed for this query

## Benefits

### For Analysis
1. **Quickly identify problematic queries**: Frames with high FP counts
2. **Understand model behavior**: Does it confuse specific segments?
3. **Quantify performance visually**: TP/FP ratio per frame

### For Papers/Presentations
1. **Show failure cases**: Frames with FPs highlighted in red
2. **Show success cases**: Frames with all TPs in green
3. **Demonstrate model limitations**: Clear visual distinction

### For Debugging
1. **Identify systematic errors**: Are FPs always from specific segments?
2. **Validate improvements**: Compare TP/FP ratios across model versions
3. **Understand edge cases**: When and why does the model fail?

## Comparison with Ground Truth

When comparing with ground truth visualization:
- **Ground truth GIF**: Shows spatially close neighbors (all "correct")
- **Prediction GIF**: Shows what model actually predicts (TP + FP)
- **Red circles** in prediction GIF = places where model fails

## Testing

A test script is provided:
```bash
python test_tp_fp_viz.py
```

This will:
1. Find a query with both TP and FP
2. Generate a single frame PNG
3. Show the TP/FP visualization clearly

## Notes

- **Distance threshold still applies**: FPs shown are within spatial distance threshold
- **Segment-based classification**: TP/FP is based on segment labels, not spatial proximity
- **Descriptor distance**: The "dist" values shown are descriptor similarities, not spatial distances
- **Label positioning**: TP/FP labels appear slightly above each predicted neighbor

## Future Enhancements (Optional)

Possible improvements:
1. Show spatial distance alongside descriptor distance
2. Add precision/recall metrics to title
3. Color-code by confidence level (e.g., darker = higher confidence)
4. Add confusion matrix overlay
5. Show ground truth neighbors for comparison

## Files Modified

1. `plot_loop_closure_pr_gif.py` - Main visualization script
2. `test_tp_fp_viz.py` - Test script for TP/FP visualization (new)
3. `README_PREDICTIONS_GIF.md` - Updated documentation
