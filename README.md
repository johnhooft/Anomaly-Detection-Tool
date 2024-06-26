# Anomaly-Detection-Tool
 A Machine Learning Framework implemented in python designed to detect anomalies in time series data using weak-supervision machine learning.

 Anomaly Detection Tool (ADT) framework utilizes 6 light weight ML algorithms in a 2 pass technique to generate weak labeled data sets and detect anomalies.

# Setup Instructions:
- Clone Repo: ```git clone https://github.com/johnhooft/Anomaly-Detection-Tool.git```
- Download Dependencies: ```pip install -r requirements.txt```

# There are two ways to utilize this framework:

## Web Dashboard:
The web dashboard allows the user to easily apply machine learning technqiues to their data, by simplying uploading your data files through the menu system. The data must either be in XML format such as produced by Nagios RRD databases, or the user can convert their data to simple CSV Timestamp, Value format and upload it. The web dashboard allows for model selection and will visualize results.

To start the web dashboard clone this repo to your system, then run:
```python3 Dashserver.py```

## Framework API:
This framework can also be utilized through an API. By cloning this repo to your system, you can utilize the api functions defined in DashDataAPI.py, mainly the apiupload function. 
    
### API Functionality:

**apiupload(algorithm ('multi', 'gen'...), filepath)**:
- Input: This function takes two arguments, the first being a string value which defines the alogrithm (must be a valid algorithm), and a string that represents the filepath to your dataset you want the framework to be applied too.

- Return: Uppon proper use, this function returns a Pandas DataFrame containing timestamp, value, and anomaly value. timestamp: Datetime value for feature, value: Numerical Value for feature, anomaly_value: associated anomaly value of feature.

- If the anomaly value is a float, it represents out of 6 what number of labeling functions considered the point to be anomalous. If the anoamly value is an integer that is 1 or 0, 1 means the point is an anomaly, 0 means it is not.

- **algorithm** Arg: While there are a multitude of models available in this framework, the two recommended ones are: "multi" which utilizes the full capacity of the framework. It summs the results of the 6 leight weight models with majority vote, and returns an anomaly_value of 0 - 6 in the dataframe. Features with an anomaly_value greater then 4 are considered to be an anomaly. The other main model is "gen" which uses the Snorkle Generative Labeling Model (https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/labeling/snorkel.labeling.LabelModel.html) to map the 6 weak labels to a stronger single label. This returns an anomaly_value of 1 or 0.

