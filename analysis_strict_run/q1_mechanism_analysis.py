import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

p = 'R3/results_strict_20260717_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260717_015946/tables/merge_mechanism_by_method_step.csv'
df = pd.read_csv(p)

pre = df[df['phase']=='pre_merge'].copy()
pre['task_step'] = pre['task_step'].astype(int)
pre_t2plus = pre[pre['task_step']>1]

print("=== PRE-MERGE: mean cos(dW1,dWt) for t=2..5, and mean dW_norm_over_dW1_norm, by method ===")
g = pre_t2plus.groupby('method').agg(
    mean_cos_t2plus=('cos_dW1_dWt','mean'),
    std_cos_t2plus=('cos_dW1_dWt','std'),
    mean_norm_ratio_t2plus=('dW_norm_over_dW1_norm','mean'),
).round(4)
print(g)
g.to_csv('analysis_strict_run/Q1_premerge_cos_and_norm_by_method.csv')

print()
print("=== POST-MERGE: cos(dW1, merged) and merged_norm_over_mean_individual_norm, by method ===")
post = df[df['phase']=='post_merge']
g2 = post.groupby('method').agg(
    mean_cos_dW1_merged=('cos_dW1_dWt','mean'),
    std_cos_dW1_merged=('cos_dW1_dWt','std'),
    mean_merged_norm_ratio=('merged_norm_over_mean_individual_norm','mean'),
    std_merged_norm_ratio=('merged_norm_over_mean_individual_norm','std'),
).round(4)
print(g2)
g2.to_csv('analysis_strict_run/Q1_postmerge_cos_and_norm_by_method.csv')

print()
print("=== per-step breakdown of pre-merge cos, orth vs non-orth pairs ===")
g3 = pre.groupby(['method','task_step']).agg(
    mean_cos=('cos_dW1_dWt','mean'),
    mean_norm_ratio=('dW_norm_over_dW1_norm','mean'),
).round(4)
print(g3)
g3.to_csv('analysis_strict_run/Q1_premerge_cos_by_method_step.csv')
