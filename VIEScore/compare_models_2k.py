import pandas as pd
import glob

files = glob.glob("2k_viescore_results_*.csv")
dfs = []

for file in files:
    df = pd.read_csv(file)
    model_name = file.split("2k_viescore_results_")[-1].split(".csv")[0]
    df[['SC', 'PQ', 'O']] = df[model_name].apply(lambda x: pd.Series(eval(x)))
    df = df.drop(columns=[model_name])
    df = df.rename(columns={'SC': f'{model_name}_SC', 'PQ': f'{model_name}_PQ', 'O': f'{model_name}_O'})
    dfs.append(df)

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.merge(df, on='input_image', how='outer')

models_metrics = [(col.rsplit('_', 1)[0], col.rsplit('_', 1)[1], merged_df[col].mean()) for col in merged_df.columns[1:]]

models_metrics_df = pd.DataFrame(models_metrics, columns=['Model', 'Metric', 'Average Score'])
models_metrics_df['Average Score'] = models_metrics_df['Average Score'].round(2)

metrics_table = models_metrics_df.pivot(index='Model', columns='Metric', values='Average Score').reset_index()
metrics_table = metrics_table.sort_values(by='O', ascending=False)  # Sort by 'O' score in descending order

metrics_table.to_csv('2k_vies_model_scores.csv', index=False)
