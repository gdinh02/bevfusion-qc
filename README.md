# Lane evaluation fixes — 0.1 m tolerance

Copy the three replacement scripts and the new `lane_evaluation.py` into the
root of your `code-cleanup` checkout. Keep its existing `lane_inference` package.
Optionally copy `test_lane_evaluation.py` too. No changes have been pushed.

Alternatively, run `git apply --check precision-fixes.patch` then
`git apply precision-fixes.patch` from the repository root. Use either the
replacement files or the patch, not both.

Run your existing batch command:

```bash
python run_batch_eval.py
```

Its dataset paths remain at the top of that file. Pixel scale must remain
consistent: `demo_gt_settings.py` and the evaluator both use 33 px/m.

Evaluate an existing pair of mask folders directly:

```bash
python eval-script.py --gt pred-dir --pred pred-dir --vehicles pred-dir --tolerance-m 0.1 --ppm 33
```

Regenerate prediction masks first: old masks used display-dependent thickness.
The GT source videos must represent the same centred 40 m by 40 m BEV and frame
indices as before; the scorer cannot verify their annotation alignment.

## What changed

- Predictions are rasterised directly from `last_boundaries`, at equal one-pixel
  width with no antialiasing, independent of boundary confidence/source styling.
- Both masks are skeletonised using Zhang-Suen thinning (NumPy/OpenCV only).
- Precision is the fraction of predicted skeleton pixels within Euclidean
  distance 0.1 m of GT. Recall reverses that direction. F1 combines these values.
- Distance is measured at the raster resolution; 0.1 m corresponds to 3.3 pixels
  at 33 px/m. Diagonal distances use Euclidean distance, not square dilation.
- Exact skeleton IoU remains under the `IoU` output label for batch compatibility;
  it does not use tolerance and cannot be inferred from the tolerant F1.
- Raw counts and the empty-denominator convention (zero) are printed.
- Missing/unreadable prediction masks or incompatible shapes fail evaluation.
  A valid frame with no predicted boundaries still writes an empty prediction.
- The batch summary records the tolerance and metric definition.

This is a point-based geometric metric, not a one-to-one lane-instance matching
metric. Nearby duplicate boundaries can both match GT. It weights raster pixels,
so it is not exactly continuous curve arc-length weighting.

## Compatibility and scope

`--kernel` now produces an explicit migration error instead of silently using
an obsolete dilation setting. Other callers such as `gt_sweep_eval.py` and
`run_batch_eval_with_model.py` must replace their `--kernel` argument with
`--tolerance-m 0.1` before using this evaluator. This package targets
`run_batch_eval.py` and its GT mask producer only.

Scene eligibility, scene failure aggregation and macro averaging remain as
before. Missing-file errors are still handled by the batch driver's existing
failed-scene skip behaviour. Check the console for failed scenes.

## Validation

```bash
python -m unittest -v test_lane_evaluation.py
```

Nine synthetic tests passed with NumPy and opencv-python-headless. All modified
Python files compile. Full nuScenes inference was not run here; it needs your
dataset and existing project dependencies. The rendering unit test isolates the
polynomial evaluator with the repository's verified ascending-coefficient
convention.

Compare configurations on the same scenes under this new metric. Earlier
dilated-mask scores are not directly comparable.
