# TMA Project DNS Tunneling Detection

### DataSet

### DataSet Preprocessing

### Random Forest Tree with XGBoost

### Structure
- /Datasets -> Set of Datasets collected
- /FeatureCollector -> Generate the .csv files from the .pcap collected data
- rf_build.py -> Training and predicting the splitted .csv samples
- rf_hyperparams.py -> Processing the RF hyperparameter
- csv_appendColumn.py -> Script utility for adding a column to an .csv file
- csv_merge.py -> Script utility for merging .csv files

### Execution
- python rf_build.py <Path/to/dataset.csv>

### Examples
- python rf_build.py featuresart.csv
- python rf_build.py featuresff.csv
