"""
Standalone, no-torch demonstration of the mechanism identified in report.txt:
an open-100-way argmax evaluation, combined with classifier rows that have
different norms depending on how recently they were trained (uncorrected when
apply_calibration=False), produces exactly the "step_5 always wins" pattern
observed in R3's per_step_accuracy_by_method.csv -- and shows that the new
restricted_argmax_accuracy() function (reimplemented verbatim here) removes
it. Pure numpy, deterministic, no dependency on the main script or any model.
"""
import numpy as np

rng = np.random.default_rng(0)

NUM_CLASSES = 100
CLASSES_PER_STEP = 20
N_STEPS = 5
N_EVAL_PER_STEP = 200  # toy eval-set size per step

# Simulate: each step's 20 classes have a classifier row whose norm reflects
# how recently it was trained -- step 1 (trained longest ago, never
# renormalized) gets the smallest row norm, step 5 (just trained) gets the
# largest, monotonically -- mirroring calibrate_classifier_row_norms()'s own
# docstring description of the uncorrected failure mode for apply_calibration
# =False methods. Directions are otherwise well-separated (each class's true
# row points at its own eval images plus small noise), so a RESTRICTED
# (closed-set) evaluation should recover near-ceiling accuracy for every step
# equally -- only the OPEN evaluation should show a step_5 bias.
row_norm_by_step = [0.3, 0.5, 0.7, 0.9, 1.6]  # step 5 much larger, uncalibrated

W = rng.normal(size=(NUM_CLASSES, 16)) * 0.05
for step in range(N_STEPS):
    classes = range(step * CLASSES_PER_STEP, (step + 1) * CLASSES_PER_STEP)
    for c in classes:
        v = rng.normal(size=16)
        W[c] = row_norm_by_step[step] * v / np.linalg.norm(v)


def restricted_argmax_accuracy(logits, labels, allowed_class_ids):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    mask = np.full(logits.shape[1], -np.inf, dtype=np.float64)
    allowed = sorted({int(c) for c in allowed_class_ids})
    mask[allowed] = 0.0
    masked_logits = logits + mask[None, :]
    preds = np.argmax(masked_logits, axis=1)
    return float((preds == labels).mean())


print(f"{'step':>4} {'open_acc':>10} {'restricted_acc':>15} {'recency_bias_gap':>18}")
for step in range(N_STEPS):
    classes = list(range(step * CLASSES_PER_STEP, (step + 1) * CLASSES_PER_STEP))
    labels = rng.choice(classes, size=N_EVAL_PER_STEP)
    # "features" for each eval image: the true class's row direction plus
    # noise (a well-separated, easily-classifiable-in-isolation image, same
    # as any converged model's own-class images should be).
    feats = np.stack([
        (W[y] / np.linalg.norm(W[y])) * row_norm_by_step[step] + rng.normal(size=16) * 0.05
        for y in labels
    ])
    logits = feats @ W.T  # [N, 100], exactly like the real classifier: features . W^T

    open_preds = np.argmax(logits, axis=1)
    open_acc = float((open_preds == labels).mean())
    restricted_acc = restricted_argmax_accuracy(logits, labels, classes)
    print(f"{step + 1:>4} {open_acc*100:>9.1f}% {restricted_acc*100:>14.1f}% {(open_acc-restricted_acc)*100:>17.1f}pp")

print()
print("Interpretation: restricted (closed-set) accuracy is near-ceiling and FLAT")
print("across all 5 steps (the underlying representation is equally good for every")
print("step's own classes) -- only the OPEN accuracy shows the step-5-biggest,")
print("step-1-smallest pattern, driven entirely by the injected row-norm schedule.")
print("This is the same mechanism calibrate_classifier_row_norms() exists to fix,")
print("and the same reason apply_calibration=False (rank_extension family) is the")
print("family whose per-step accuracy swing is most extreme in R3's real numbers.")
