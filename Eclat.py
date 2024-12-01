import pandas as pd

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 5 - Association Rule Learning\Section 28 - Apriori\Python\Market_Basket_Optimisation.csv",header=None)
df = df.astype(str)


from pyECLAT import ECLAT
eclat_instance = ECLAT(data=df)
indexes, supports = eclat_instance.fit(min_support=.003,min_combination=2,max_combination=2)
print(indexes)
print(supports)