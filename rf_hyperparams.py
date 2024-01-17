import pandas as pd
import numpy as np

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split

import multiprocessing, warnings, sys

# features = pd.read_csv('featuresf.csv')\
    # .get(["rr_name_entropy", "rr_name_length", "malicious"])
# features = pd.read_csv('featuresff.csv')\
features = pd.read_csv(sys.argv[1])\
    # .get(["subdomain_length", "numeric", "entropy", "len", "malicious"])

# print(features.head(5))

# labels = np.array(features['malicious'])
labels = features['malicious']
features = features.drop('malicious', axis = 1)
features = features.drop('timestamp', axis = 1)
try:
    features = features.drop('time_difference', axis = 1)
except:
    features = features.drop('sld', axis = 1)
    features = features.drop('longest_word', axis = 1)

feature_list = list(features.columns)
# features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 123)

# Out-of-bag Score

param_grid = ParameterGrid(
                {'n_estimators': [
                    50,
                    100,
                    150],
                 'max_features': [5, 7, 9],
                 'max_depth'   : [None, 3, 10, 20],
                 'criterion'   : ['gini', 'entropy']
                }
            )

# Concurrently adjust hyperparams
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

def get_hparams():
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
    print(results.head(5))
    return {"n_estimators": results.iloc[0]['n_estimators'], 
            "max_features": results.iloc[0]['max_features'],
            "max_depth": 0 if pd.isna(results.iloc[0]['max_depth']) else int(results.iloc[0]['max_depth']),
            "criterion": results.iloc[0]['criterion']}

if __name__ == '__main__':
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
    # print(results.iloc[0]['n_estimators'], results.iloc[0]['max_features'])