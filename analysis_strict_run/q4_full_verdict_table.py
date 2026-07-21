import pandas as pd
pd.set_option('display.width', 250)

paths = {
    'OLD': 'R3/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260715_014445/tables/final_metrics_all_methods.csv',
    'CALIBFIX': 'R3/results_calibfix_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_133338/tables/final_metrics_all_methods.csv',
    'REVERT': 'R3/results_revert_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_193538/tables/final_metrics_all_methods.csv',
    'STRICT': 'R3/results_strict_20260717_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260717_015946/tables/final_metrics_all_methods.csv',
}

frames = []
for run, p in paths.items():
    d = pd.read_csv(p)
    d['run'] = run
    frames.append(d[['run', 'method', 'first_step_accuracy', 'later_steps_accuracy', 'all_seen_accuracy', 'forgetting_metric', 'backward_transfer']])
long = pd.concat(frames)
long.to_csv('analysis_strict_run/Q4_raw_metrics_all_runs_long.csv', index=False)

wide = long.pivot(index='method', columns='run')
wide.columns = [f'{m}_{r}' for m, r in wide.columns]
wide = wide.reset_index()

for metric in ['all_seen_accuracy', 'first_step_accuracy', 'later_steps_accuracy']:
    wide[f'delta_{metric}_STRICT_minus_REVERT'] = wide[f'{metric}_STRICT'] - wide[f'{metric}_REVERT']
    wide[f'delta_{metric}_STRICT_minus_OLD'] = wide[f'{metric}_STRICT'] - wide[f'{metric}_OLD']

col_order = ['method']
for metric in ['all_seen_accuracy', 'first_step_accuracy', 'later_steps_accuracy', 'forgetting_metric', 'backward_transfer']:
    for run in ['OLD', 'CALIBFIX', 'REVERT', 'STRICT']:
        col_order.append(f'{metric}_{run}')
for metric in ['all_seen_accuracy', 'first_step_accuracy', 'later_steps_accuracy']:
    col_order.append(f'delta_{metric}_STRICT_minus_REVERT')
    col_order.append(f'delta_{metric}_STRICT_minus_OLD')

wide = wide[col_order]
wide.to_csv('analysis_strict_run/Q4_full_per_method_table_4runs.csv', index=False)
print(wide.round(2).to_string(index=False))
