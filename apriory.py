import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 5 - Association Rule Learning\Section 28 - Apriori\Python\Market_Basket_Optimisation.csv", header=None)
df = df.astype(str)
data = df.values
print(len(data))

from apyori import apriori
rules = apriori(transactions=data,min_support = (3*7)/7501, min_confidence=20/100, min_lift=3,min_length=2)

li = (list(rules))

for i in li:
    print(list(i))


