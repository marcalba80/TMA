import pandas as pd
import numpy as np

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Tree Visualisation
import matplotlib.pyplot as plt
import graphviz

# features = pd.read_csv('stateful_features-light_text.pcap.csv')\
features = pd.read_csv('featuresf.csv')\
    # .get(["rr_name_entropy", "rr_name_length", "malicious"])
# print(features)
# print(features.head(5))

labels = np.array(features['malicious'])
features = features.drop('malicious', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 123)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestClassifier(n_estimators=150, max_depth=3, max_features=5, criterion="gini")
rf.fit(train_features, train_labels)

test_pred = rf.predict(test_features)

accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred)
recall = recall_score(test_labels, test_pred)
# print("Accuracy: ", accuracy)

mat_confusion = confusion_matrix(
                    y_true    = test_labels,
                    y_pred    = test_pred
                )

print("-------------------")
print("Confusion Mat")
print("-------------------")
print(mat_confusion)
print("")
print(f"Accuracy: {100 * accuracy} %")
print(f"Precision: {100 * precision} %")
print(f"Recall: {100 * recall} %")

fig, ax = plt.subplots(figsize=(3, 3))
ConfusionMatrixDisplay(mat_confusion).plot(ax=ax)
plt.show()