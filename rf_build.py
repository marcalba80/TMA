import pandas as pd
import numpy as np

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRFClassifier, DMatrix

# Tree Visualisation
import matplotlib.pyplot as plt
import graphviz

from rf_hyperparams import get_hparams

import os, warnings

features = pd.read_csv('featuresf.csv')\
    .get(["rr_name_entropy", "rr_name_length", "malicious"])
# features = pd.read_csv('featuresf_stateless_red.csv')
    # .get(["subdomain_length", "numeric", "entropy", "len", "malicious"])
    
# print(features)
# print(features.head(5))

# labels = np.array(features['malicious'])
labels = features['malicious']
features = features.drop('malicious', axis = 1)
feature_list = list(features.columns)
# features = np.array(features)

# warnings.filterwarnings('ignore')

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 123)

cat_cols = train_features.select_dtypes(include=['object', 'category']).columns.to_list()
numeric_cols = train_features.select_dtypes(include=['float64', 'float', 'int']).columns.to_list()

preprocessor = ColumnTransformer(
                    [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), cat_cols)],
                    remainder='passthrough',
                    verbose_feature_names_out=False
               ).set_output(transform="pandas")

train_features_prep = preprocessor.fit_transform(train_features)
test_features_prep  = preprocessor.transform(test_features)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
train_features_prep.info()

# os._exit(1)

params = get_hparams()
# rf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#                             max_depth=params['max_depth'], 
#                             max_features=params['max_features'], 
#                             criterion=params['criterion'])
rf = XGBRFClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            # max_features=params['max_features'], 
            # criterion=params['criterion'],
            enable_categorical=True,
            learning_rate=1,
            subsample=0.8,
            colsample_bynode=0.8,
            random_state=123)
rf.fit(train_features_prep, train_labels)

test_pred = rf.predict(test_features_prep)

# print(test_labels)
# print(test_pred)

accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred)
recall = recall_score(test_labels, test_pred)
print(classification_report(test_labels, test_pred))

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