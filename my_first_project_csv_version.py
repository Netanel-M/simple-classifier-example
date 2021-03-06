#!/usr/bin/python3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
from sklearn import preprocessing

raw_data = pd.read_csv("female_male_weights_heights.csv", sep=",")

data = raw_data[["Weight", "Height", "Hair", "Nails"]].to_numpy()
genders = raw_data.Gender.to_numpy()


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(["Female", "Male"])

targets = label_encoder.transform(
    genders
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data, targets, random_state=0)

# Initiate the classifier algorithm
#classifier = KNeighborsClassifier(n_neighbors=1)
#classifier = GaussianNB()
#classifier = MultinomialNB(alpha=0.2)
#classifier = BernoulliNB(alpha=0.2)
#classifier = DecisionTreeClassifier(random_state=0, max_depth=8)
classifier = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])

classifier.fit(X_train, y_train)

query = {
    "height": 1.4,
     "weight": 50,
     "hair": 40,
     "nails": 4
    }
    #  42	1.5	35	4

prediction = classifier.predict(np.array([[query["weight"], query["height"], query["hair"],query["nails"]]]))


print(label_encoder.inverse_transform(prediction))
print("Test set score: {:.2f}".format(classifier.score(X_test, y_test)))
#export_graphviz(classifier, out_file="tree.dot", class_names=["Female", "Male"],
#feature_names=["Weight", "Height", "Hair", "Nails"], impurity=False, filled=True)

#import graphviz
#with open("tree.dot") as f:
#    dot_graph = f.read()
#print(graphviz.Source(dot_graph).view())