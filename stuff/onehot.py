import pandas as pd
import numpy as np

df = pd.read_excel("/Users/jonathan.bach/Documents/UNI/DataLiteracy/Allgemeine Tabelle + Prog Kopie.xlsx")

print(df.head())

df_encoded = pd.get_dummies(df, columns=['Krankheit'], prefix='Krankheit')

print(df_encoded.head())

df_encoded = df_encoded.astype(int)

df_encoded.to_excel("kitrainINT.xlsx", index=False)