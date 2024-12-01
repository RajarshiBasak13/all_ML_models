import random

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 6 - Reinforcement Learning\Section 32 - Upper Confidence Bound (UCB)\Python\Ads_CTR_Optimisation.csv")
data = df.values

D = 10
N= 10000
ads_selected = []
no_of_reward_1 = [0]*D
no_of_reward_0 = [0]*D
total_reward = 0
for n in range(N):
    max_random_beta = 0
    ad = 0
    for d in range(D):
        random_beta = random.betavariate(no_of_reward_1[d]+1,no_of_reward_0[d]+1)
        if(max_random_beta < random_beta):
            max_random_beta = random_beta
            ad = d
    print(max_random_beta, ad)
    ads_selected.append(ad)
    reward = data[n][ad]
    if(reward == 0):
        no_of_reward_0[ad] += 1
    else:
        no_of_reward_1[ad] += 1
    total_reward += reward
print(total_reward)

plt.hist(ads_selected)
plt.title("ads_clicked")
plt.xlabel("ads")
plt.ylabel("no of times watched")
plt.show()


