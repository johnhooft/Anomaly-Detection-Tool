import anomalyModels as model
import parsedata
import numpy as np
import math
import pandas as pd

UPLOAD_DIRECTORY = "/useruploads"
path = '/Users/johnhoofttoomey/OHAZ/Nagios_Data/data/' # Put your path to data here
data = ["bendodot_router", "bendodot_router_training", "wagontire_router7_rx", "wagontire_router7_tx",
        "wagontire_router4_rx", "wagontire_router4_tx", "bryant_edge_rx", "bryant_edge_tx", "blue_edge_rtt", 
        "bryant_hog_radio_rtt", "bryant_edge_rv55_rtt"]


def main(algorithm, dataset, data):
    contamination = 0
    file = data[dataset-1]

    timestamps, values, t, v = parsedata.parse_xml_data(file, path)
    datastack = np.column_stack((timestamps, values))

    if algorithm == None:
        ValueError("No algorithm selected")

    if algorithm == 'multi':
        anomaly_indices, k, n_clusters, contamination = model.multi(values=values, timestamps=timestamps, file=file)

    elif algorithm == 'gen':
        anomaly_indices = model.snorkleModel(values=values, timestamps=timestamps, file=file)
        
    elif algorithm == 'dbscan':
        eps = model.calculate_eps(values, timestamps)
        anomaly_indices, contamination = model.dbscan(datastack, eps, min_samples=3)
        print("\nCONTAMINATION = ", contamination)
        
    elif algorithm == 'svm':
        con = model.calculate_contamination(values, timestamps)
        anomaly_indices, contamination = model.one_class_svm(datastack, con)
        print("\nCONTAMINATION = ", contamination)
        
    elif algorithm == 'lof':
        k = math.floor(math.sqrt(len(values)))
        contamination = model.calculate_contamination(values, timestamps)
        anomaly_indices = model.lof(datastack, k, contamination)
        print("\nCONTAMINATION = ", contamination)
        
    elif algorithm == 'knn':
        k = math.floor(math.sqrt(len(values)))
        anomaly_indices, contamination = model.knn(datastack, k)
        print("\nCONTAMINATION = ", contamination)
        
    elif algorithm == "iso":
        con = model.calculate_contamination(values, timestamps)
        anomaly_indices = model.iso(datastack, con)
        print("\nCONTAMINATION = ", con)
        
    elif algorithm == "elip":
        con = model.calculate_contamination(values, timestamps)
        anomaly_indices = model.elip_env(datastack, contamination=con)
        print("\nCONTAMINATION = ", con)

    else:
        ValueError("No algorithm found")

    return t, v, anomaly_indices, contamination

def apiupload(algorithm, filepath):
    filetype = filepath.split(".")[1]
    if filetype == "xml":
        timestamps, values, t, v = parsedata.parse_binary_xml_data(filepath)
    elif filetype == "csv":
        timestamps, values, t, v = parsedata.parse_binary_csv(filepath)
    else:
        raise ValueError("Unsuported File Type")

    if algorithm == 'multi':
        anomaly_indices, k, n_clusters, contamination = model.multi(values=values, timestamps=timestamps, file=None)
    else:
        anomaly_indices = model.snorkleModel(values, timestamps, file=None)
        contamination = 0

    if len(values) != len(anomaly_indices):
        temp = [0] * len(values)
        for index in anomaly_indices:
            temp[index] = 1
        anomaly_indices = temp

    df = pd.DataFrame({
        'timestamp': t,
        'value': v,
        'anomaly_value': anomaly_indices
    })

    metadf = pd.DataFrame({
        'con_val': [contamination]
    })

    return df, metadf

def apimain(algorithm, dataset):
    datasetnum = -1
    for i in range(len(data)):
        if data[i] == dataset:
            datasetnum = i
    datasetnum += 1
    if datasetnum == -1:
        print("error occured with dataset")
        raise ValueError
    timestamps, values, anomaly_indices, contamination = main(algorithm, datasetnum, data)

    if len(values) != len(anomaly_indices):
        temp = [0] * len(values)
        for index in anomaly_indices:
            temp[index] = 1
        anomaly_indices = temp

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'anomaly_value': anomaly_indices
    })

    metadf = pd.DataFrame({
        'con_val': [contamination]
    })

    return df, metadf