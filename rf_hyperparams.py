import pandas as pd
import numpy as np

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid, train_test_split
# from scipy.stats import randint

# # Tree Visualisation
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import graphviz

import multiprocessing, warnings

# features = pd.read_csv('stateful_features-light_text.pcap.csv')\
features = pd.read_csv('featuresf.csv')\
    .get(["rr_name_entropy", "rr_name_length", "malicious"])
# print(features)
# print(features.head(5))

# Labels are the values we want to predict
labels = np.array(features['malicious'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('malicious', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 123)

# Out-of-bag Score

param_grid = ParameterGrid(
                {'n_estimators': [150],
                 'max_features': [5, 7, 9],
                 'max_depth'   : [None, 3, 10, 20],
                 'criterion'   : ['gini', 'entropy']
                }
            )

# Loop adjust hyperparams
# ==============================================================================
def eval_oob_error(X, y, modelo, params, verbose=True):
    modelo.set_params(
        oob_score    = True,
        n_jobs       = -1,
        random_state = 123,
        ** params            
    )
    
    modelo.fit(X, y)
        
    return{'params': params, 'oob_accuracy': modelo.oob_score_}

warnings.filterwarnings('ignore')
n_jobs     = multiprocessing.cpu_count() -1
pool       = multiprocessing.Pool(processes=n_jobs)
results = pool.starmap(
                eval_oob_error,
                [(train_features, train_labels, RandomForestClassifier(), params) for params in param_grid]
             )
warnings.filterwarnings('default')

results = pd.DataFrame(results)
results = pd.concat([results, results['params'].apply(pd.Series)], axis=1)
results = results.drop(columns = 'params')
results = results.sort_values('oob_accuracy', ascending=False)
print(results.head(4))