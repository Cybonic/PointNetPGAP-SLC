# Plot Updates - PDF Output with No Axes

## Changes Made

### ✅ Removed all axes elements
- `ax.set_axis_off()` - Removes all axis lines, ticks, and labels
- Background panes set to transparent
- Grid completely disabled
- Only the 3D plot data remains visible

### ✅ PDF Output
- Files now saved as **PDF** format (vector graphics)
- Changed from `.png` to `.pdf` extension
- Better quality for publications
- Smaller file sizes
- Scalable without quality loss

### ✅ Tight Fitting
- `bbox_inches='tight'` - Removes extra whitespace
- `pad_inches=0` - No padding around the plot
- Figure is fitted tightly to the 3D visualization

### ✅ Optional Title
- Title shown only when `show_legend=True`
- Displays: `{model_name} - {sequence}` and `True Positives: {count}`
- For completely clean plots, set `show_legend=False`

## Output Format

### File Names
```
{sequence}_{model}_tp.pdf
```

Examples:
- `PCD_MED_PointNetPGAP_tp.pdf`
- `PCD_Easy_DARK_PointNetVLAD_tp.pdf`

### Visual Style
- **Black trajectory path** (no axes, no grid, no background)
- **Green loop closure connections**
- **White background** (for clean integration in documents)
- **Vector format** (PDF) for high-quality publications

## Configuration

### Minimal Clean Style (Recommended)
```python
show_grid = False      # No grid
show_legend = False    # No title or legend
show_axes = False      # Not used (axes always off now)
```

**Result**: Pure 3D visualization - only black path and green connections

### With Title
```python
show_legend = True     # Shows title with model name and TP count
```

**Result**: Same clean plot with title at top

## Benefits of PDF Format

1. **Vector Graphics**: Infinite zoom without pixelation
2. **Small File Size**: More efficient than high-DPI PNG
3. **Publication Ready**: Journals prefer vector formats
4. **Easy Integration**: Works seamlessly in LaTeX, Word, PowerPoint
5. **Editable**: Can be further edited in Adobe Illustrator or Inkscape

## File Size Comparison

- **PNG (300 DPI)**: ~1-2 MB per file
- **PDF (vector)**: ~100-500 KB per file
- **Result**: ~75% smaller files with better quality

## Usage Example

```bash
cd /home/tiago/workspace/place_uk/PointNetGAP
python dataloader/unit_test/plot_tp_individual.py
```

Output:
```
plots/true_positives_individual/
├── PCD_Easy_DARK_PointNetPGAP_tp.pdf     (367 TPs)
├── PCD_Easy_DARK_PointNetVLAD_tp.pdf     (347 TPs)
├── PCD_Easy_DARK_SPVSoAP3D_tp.pdf        (299 TPs)
├── PCD_Easy_DARK_LOGG3D_tp.pdf           (234 TPs)
├── PCD_Easy_DARK_overlap_transformer_tp.pdf (217 TPs)
├── PCD_MED_PointNetPGAP_tp.pdf           (523 TPs)
├── PCD_MED_PointNetVLAD_tp.pdf           (461 TPs)
├── PCD_MED_SPVSoAP3D_tp.pdf              (337 TPs)
├── PCD_MED_LOGG3D_tp.pdf                 (185 TPs)
└── PCD_MED_overlap_transformer_tp.pdf    (152 TPs)
```

## LaTeX Integration

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.45\textwidth]{plots/PCD_MED_PointNetPGAP_tp.pdf}
    \caption{True positive loop closures for PointNetPGAP on PCD\_MED sequence.}
    \label{fig:tp_pnpgap}
\end{figure}
```

## Multi-Panel Figures in LaTeX

```latex
\begin{figure*}[t]
    \centering
    \begin{subfigure}{0.19\textwidth}
        \includegraphics[width=\textwidth]{plots/PCD_MED_PointNetPGAP_tp.pdf}
        \caption{PointNetPGAP}
    \end{subfigure}
    \begin{subfigure}{0.19\textwidth}
        \includegraphics[width=\textwidth]{plots/PCD_MED_PointNetVLAD_tp.pdf}
        \caption{PointNetVLAD}
    \end{subfigure}
    \begin{subfigure}{0.19\textwidth}
        \includegraphics[width=\textwidth]{plots/PCD_MED_SPVSoAP3D_tp.pdf}
        \caption{SPVSoAP3D}
    \end{subfigure}
    \begin{subfigure}{0.19\textwidth}
        \includegraphics[width=\textwidth]{plots/PCD_MED_LOGG3D_tp.pdf}
        \caption{LOGG3D}
    \end{subfigure}
    \begin{subfigure}{0.19\textwidth}
        \includegraphics[width=\textwidth]{plots/PCD_MED_overlap_transformer_tp.pdf}
        \caption{OverlapTF}
    \end{subfigure}
    \caption{True positive loop closures comparison across five models on PCD\_MED sequence.}
    \label{fig:tp_comparison}
\end{figure*}
```

## Tips

1. **Consistent Viewpoint**: Keep `view_elev` and `view_azim` the same for all plots
2. **Clean Style**: Use `show_legend=False` for multi-panel figures (add labels in LaTeX)
3. **Vector Format**: PDFs scale perfectly at any size
4. **White Background**: Works with both white and colored document backgrounds

## What Changed in Code

### Before (PNG with axes):
```python
# Had axis labels and ticks
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
# Saved as PNG
plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### After (PDF without axes):
```python
# Remove all axes
ax.set_axis_off()
# Clean background
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
# Save as PDF with tight fit
plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
```

## Result

Clean, publication-ready PDF files showing only:
- Black trajectory path
- Green loop closure connections
- Optional title (if show_legend=True)
- Perfect for papers, presentations, and posters!
