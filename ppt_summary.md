# n5_methods_explanation.pptx summary

Source notebook: `vit_lora_cifar100_full5step_n5.ipynb`

## Slide-to-notebook mapping

1. **Title**
- Notebook scope only.

2. **Experimental Setup and Notation**
- Cell 3: full comparison config (`NUM_STEPS`, `CLASSES_PER_STEP`, `TARGET_MODULES`, `METHODS_TO_RUN`, orth/rankext hyperparameters).

3. **Simple Average Merging (Concept)**
- Cell 11: `train_independent_loras(...)`.
- Cell 12: `simple_avg_no_replay`, `simple_avg_orth` blocks.
- Cell 13: `simple_avg_replay` block.

4. **Simple Average Formula + Code**
- Cell 9: `simple_average_deltas(step_states)`.
- Cell 12/13: `simple_delta`, `replay_delta`, `apply_deltas_to_base(...)`.

5. **Rank Extension (Concept)**
- Cell 16: `GrowingRankLoRALinear` design and high-level behavior.

6. **Rank Extension Math + Layer Construction**
- Cell 16: `GrowingRankLoRALinear.full_A_B`, `GrowingRankLoRALinear.forward`.
- Cell 16: `build_rank_extension_model(previous_rank_state, step_idx)`.

7. **Rank Extension Training + Diagnostics**
- Cell 16: `run_rank_extension_variant(...)`.
- Cell 16: `snapshot_frozen_rank_blocks(...)`, `check_frozen_rank_blocks_unchanged(...)`, `extract_rank_extension_state(...)`.
- Cell 16: `rank_extension_variants` list.

8. **DO-Merging (Concept)**
- Cell 9: `column_decouple_delta`, `orthogonalize_task_directions`, `do_merge_deltas`.
- Cell 14: `do_merging_simple`/`do_merging_simple_orth` execution blocks.

9. **DO-Merging Formula + Code**
- Cell 9: `do_merge_deltas(...)` and `apply_deltas_to_base(...)`.

10. **Orthogonality / Orthogonal Loss (Concept)**
- Cell 15: equation comments and orth objective framing.

11. **Orthogonality Code Path + Hyperparameters**
- Cell 15: `extract_reference_weights_for_orth`, `compute_orth_penalty`, `OrthogonalLossTrainer`.
- Cell 11: `IndependentLoraOrthTrainer` path in `train_independent_loras(..., use_orth=True)`.
- Cell 16: orth path inside `run_rank_extension_variant(..., use_orth=True)`.
- Cell 3: `LAMBDA_ORTH`, `ORTH_SCALE_MODE`, `ORTH_TARGET_RATIO`, `ORTH_LAMBDA_MIN/MAX`, `ORTH_EPS`, `ORTH_DIAGNOSTICS`.

12. **Comparison Framework and Metrics**
- Cell 19: final pivot/summary (`first_step`, `later_steps`, `all_seen`, replay gains, orth gains, rank-extension gains, gap to joint upper bound).

## Notes

- The presentation was created by reading/analyzing the notebook only.
- No modifications were made to `vit_lora_cifar100_full5step_n5.ipynb`.
