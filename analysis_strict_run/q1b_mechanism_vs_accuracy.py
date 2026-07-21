import pandas as pd
pd.set_option('display.width', 200)

STRICT = 'R3/results_strict_20260717_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260717_015946'
mech = pd.read_csv(f'{STRICT}/tables/merge_mechanism_by_method_step.csv')
acc = pd.read_csv(f'{STRICT}/tables/final_metrics_all_methods.csv')

post = mech[mech.phase == 'post_merge'].groupby('method').agg(
    mean_cos_dW1_merged=('cos_dW1_dWt', 'mean'),
    mean_merged_norm_over_mean_individual=('merged_norm_over_mean_individual_norm', 'mean'),
).reset_index()

mech['task_step_num'] = pd.to_numeric(mech.task_step, errors='coerce')
pre = mech[(mech.phase == 'pre_merge') & (mech.task_step_num > 1)].groupby('method').agg(
    mean_cos_t2plus_pre_merge=('cos_dW1_dWt', 'mean'),
    mean_dWnorm_over_dW1norm_t2plus=('dW_norm_over_dW1_norm', 'mean'),
).reset_index()

simple_avg_methods = ['simple_avg', 'simple_avg_kd_T2', 'simple_avg_factor_orth', 'simple_avg_factor_orth_kd_T2']
a = acc[acc.method.isin(simple_avg_methods)][['method', 'first_step_accuracy', 'later_steps_accuracy', 'all_seen_accuracy']]

out = a.merge(pre, on='method').merge(post, on='method')
out = out.sort_values('first_step_accuracy', ascending=False)
print(out.round(4).to_string(index=False))
out.to_csv('analysis_strict_run/Q1_mechanism_vs_accuracy_STRICT_run.csv', index=False)

print()
print("Correlation (n=4 simple_avg variants, STRICT run):")
print("first_step_accuracy vs mean_cos_dW1_merged (post-merge):", out['first_step_accuracy'].corr(out['mean_cos_dW1_merged']).round(4))
print("first_step_accuracy vs mean_merged_norm_over_mean_individual (post-merge):", out['first_step_accuracy'].corr(out['mean_merged_norm_over_mean_individual']).round(4))
print("first_step_accuracy vs mean_cos_t2plus_pre_merge:", out['first_step_accuracy'].corr(out['mean_cos_t2plus_pre_merge']).round(4))
print("first_step_accuracy vs mean_dWnorm_over_dW1norm_t2plus:", out['first_step_accuracy'].corr(out['mean_dWnorm_over_dW1norm_t2plus']).round(4))
