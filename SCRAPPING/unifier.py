import pandas as pd

df1 = pd.read_csv("./assets/total_01.csv")
df2 = pd.read_csv("./assets/total_02.csv")
df3 = pd.read_csv("./assets/total_03.csv")

min_len = min(len(df1), len(df2), len(df3))

df1 = df1.reindex(range(len(df2)))
df3 = df3.reindex(range(len(df2)))

df2['description_full'] = [
    next((v for v in vals if pd.notnull(v)), None)
    for vals in zip(df1['description_full'], df2['description_full'], df3['description_full'])
]

df2.to_csv('./assets/resultado.csv', index=False)

total_filas = len(df2)
num_con_valor = df2['description_full'].notna() & (df2['description_full'].str.strip() != "")
print(f"Total de filas: {total_filas}")
print(f"Filas con 'description_full' no vacío: {num_con_valor.sum()}")
