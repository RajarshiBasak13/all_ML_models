import nltk
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Python\Restaurant_Reviews.tsv", delimiter="\t",quoting=3)
review = data["Review"][0]

#PRE-PROCSSING

import re;
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_preprocess(row):
    review = row["Review"]
    review_with_lower_character = re.sub("[^a-zA-Z]", " ", review).lower()

    ps = PorterStemmer()
    review_with_lower_character_splitted = review_with_lower_character.split()
    review_with_lower_character_splitted_important_words = [ps.stem(word) for word in review_with_lower_character_splitted if word not in stopwords.words("english")]

    review_with_lower_character_important_words = " ".join(review_with_lower_character_splitted_important_words)
    row["Review"] = review_with_lower_character_important_words
    return row

data = data.apply(lambda x : text_preprocess(x),axis=1)
x = data["Review"].values
y = data["Liked"].values

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(x).toarray()


#classification algorithm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


























