import pandas as pd
df = pd.read_csv('mc_recovery_results.csv')
print(df[['f_stat_classical', 'f_stat_cluster']].head(10))
print()
print('Они полностью идентичны во всех 100 строках?', (df['f_stat_classical'] == df['f_stat_cluster']).all())
