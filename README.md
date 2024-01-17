# TMA Project DNS Tunneling Detection

DataSet

DataSet Preprocessing

Random Forest Tree with XGBoost

Structure
/Datasets
/FeatureCollector -> Generates the .csv files from the .pcap collected data
rf_build.py -> Training and predicting the splitted .csv samples

Execution
python rf_build.py <Path:dataset.csv>

Examples
python rf_build.py featuresart.csv
python rf_build.py featuresff.csv