# TMA Project DNS Tunneling Detection

DataSet

DataSet Preprocessing

Random Forest Tree implementation
Using the following features:
1. record name length
2. entropy of resource record name
3. unique country
4. ratio of repeated characters
5. probability from CNN(Convolutional Neural Network)
6.

Structure
/Datasets
/FeatureCollector -> Generates the .csv files from the .pcap collected data
rf_build.py -> Training and predicting the splitted .csv samples

Execution
python rf_build.py <Path:dataset.csv>

Examples
python rf_build.py featuresart.csv
python rf_build.py featuresff.csv