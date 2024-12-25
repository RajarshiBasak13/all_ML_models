import pandas as pd
import matplotlib.pyplot as plt

#gods sample, only god knows who will choose what ads, we have such god result for 10,000 diff people.
df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 6 - Reinforcement Learning\Section 32 - Upper Confidence Bound (UCB)\Python\Ads_CTR_Optimisation.csv")

#-------------------------------------------random choice result.
# import random
# d = 10
# n = 10000
# ads_selected = []
# reward_result = 0;
# for i in range(n):
#     option = random.randrange(d)
#     ads_selected.append(option)
#     reward_round = df.values[i][option]
#     reward_result += reward_round
#
# print(reward_result)
# plt.hist(ads_selected)
# plt.title("Ads_selected")
# plt.xlabel("ads")
# plt.ylabel("count")
# plt.show()

#here result of reward is around 1200.Lets see how we can improve it with UCB
#Upper Confidence Bound Algorithm
import math

data = df.values
D = 10
N = 10000
ads_selected = []
ads_showed = [0]*D
reward_by_ads = [0]*D
for n in range(N):
    upper_bound = 0
    max_upper_bound = 0
    ads = 0
    for d in range(D):
        if ads_showed[d]>0:
            avarage_reward = reward_by_ads[d] / ads_showed[d]
            delta = math.sqrt(3/2*( math.log(n)/ads_showed[d] ))
            upper_bound = avarage_reward + delta
        else:
            upper_bound = math.inf
        if max_upper_bound<upper_bound:
            ads = d
            max_upper_bound = upper_bound
    ads_selected.append(ads)
    ads_showed[ads] += 1
    reward_by_ads[ads] += data[n][ads]
print(ads_selected)

plt.hist(ads_selected)
plt.title("clicked_ads")
plt.xlabel("Ads")
plt.ylabel("No of clicks")
plt.show()
