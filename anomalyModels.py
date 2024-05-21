import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
import pandas as pd
from kneed import KneeLocator

def conv2(array):
    new_array = []
    for i in range(len(array)):
        if array[i] == -1:
            new_array.append(i)
    return new_array

def convert(array, n):
    new_array = [1] * n
    for value in array:
        new_array[value] = int(-1)
    return new_array

def calculate_contamination(values, timestamps):
    datastack = np.column_stack((timestamps, values))
    k = math.floor(math.sqrt(len(values)))
    _, knn_con = knn(datastack, k)
    eps = calculate_eps(values, timestamps)
    _, dbscan_con = dbscan(datastack, eps, min_samples=3)
    con = (knn_con + dbscan_con) / 2
    print("DBSCAN + KNN Contamination = ", con)
    return con

def calculate_eps(values, timestamps, plot=False):
    k = math.floor(math.sqrt(len(values)))
    data = np.column_stack((timestamps, values))
    knn = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = knn.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    knee_point = KneeLocator(
        range(len(distances)), distances, curve='convex')
    
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(distances)
        plt.axvline(knee_point.knee, color='blue')
        plt.title('K-distance Graph',fontsize=20)
        plt.xlabel('Data Points sorted by distance',fontsize=14)
        plt.ylabel('Epsilon',fontsize=14)
        plt.show()

    return knee_point.knee_y

def generate_label_matrix(values, timestamps):
    datastack = np.column_stack((timestamps, values))

    #KNN Labels
    k = math.floor(math.sqrt(len(values)))
    n_clusters = None
    knn_anomaly, knn_con = knn(datastack, k)

    #DBSCAN Labels
    eps = calculate_eps(values, timestamps)
    dbscan_anomaly, dbscan_con = dbscan(datastack, eps, min_samples=3)

    #Calculate Contamination Value for 2nd Pass
    con = (knn_con + dbscan_con) / 2
    print("1st Pass Contamination = ", con)

    #Remaining 2nd Pass Labels
    iso_anomaly = iso(datastack, con)
    elip_anomaly = elip_env(datastack, con)
    lof_anomaly = lof(datastack, k, con)
    svm_anomaly, svm_con = one_class_svm(datastack, con)

    # Instead of totalling up the anomaly occurences for each data point, construct 
    # Matrix instead of data points X labeling functions.
    lof_conv = convert(lof_anomaly, len(values))
    knn_conv = convert(knn_anomaly, len(values))
    dbscan_conv = convert(dbscan_anomaly, len(values))
    iso_conv = convert(iso_anomaly, len(values))
    elip_conv = convert(elip_anomaly, len(values))
    svm_conv = convert(svm_anomaly, len(values))

    # Weak Label Matrix for Training
    L = np.array([lof_conv, knn_conv, iso_conv, elip_conv, svm_conv, dbscan_conv])
    L = L.transpose()
    print(L.shape)

    # Other useful return vals: k, n_clusters, con
    return L

def snorkleModel(values, timestamps, file):

    L = generate_label_matrix(values, timestamps)

    print("Conflict / Coverage / Overlap: ")
    print(LFAnalysis(L).label_coverage())

    # Train LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L, n_epochs=1000, lr=0.001, l2=0.01, log_freq=100, seed=123, optimizer = 'adam')
    y_preds, prob = label_model.predict(L, return_probs=True)

    anomaly_indeces = conv2(y_preds)

    print(anomaly_indeces)

    return anomaly_indeces

def multi(values, timestamps, file):
    datastack = np.column_stack((timestamps, values))
    k = math.floor(math.sqrt(len(values)))
    n_clusters = None
    #n_clusters, ss = calculate_WSS(values, timestamps, graph=False)
    knn_anomaly, knn_con = knn(datastack, k)
    #_, kmean_anomaly, kmean_con = kmean2d(values, timestamps, n_clusters)
    eps = calculate_eps(values, timestamps)
    dbscan_anomaly, dbscan_con = dbscan(datastack, eps, min_samples=3)
    con = (knn_con + dbscan_con) / 2
    print("1st Pass Contamination Value = ", con)
    iso_anomaly = iso(datastack, con)
    elip_anomaly = elip_env(datastack, con)
    lof_anomaly = lof(datastack, k, con)
    svm_anomaly, svm_con = one_class_svm(datastack, con)
    
    anomalies = [lof_anomaly, knn_anomaly, dbscan_anomaly, iso_anomaly, elip_anomaly, svm_anomaly]

    # Count the occurrences of each data point index
    combined_anomalies = np.zeros(len(values))
    for anomaly in anomalies:
        unique_indices, counts = np.unique(anomaly, return_counts=True)
        combined_anomalies[unique_indices] += counts

    return combined_anomalies, k, n_clusters, con

def dbscan(data, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    predictions = model.fit_predict(data)
    anomaly_indices = np.where(predictions == -1)[0]
    con = len(anomaly_indices) / len(data)
    return anomaly_indices, con

def elip_env(data, contamination):
    model = EllipticEnvelope(contamination=contamination)
    predictions = model.fit_predict(data)
    anomaly_indices = np.where(predictions == -1)[0]
    return anomaly_indices

def one_class_svm(data, nu):
    best_pred = OneClassSVM(gamma='auto', nu=nu).fit_predict(data)
    anomaly_indices = np.where(best_pred == -1)[0]
    con = len(anomaly_indices) / len(data)
    return anomaly_indices, con

def knn(data, k):
    # Create and fit the k-NN model
    knn = NearestNeighbors(n_neighbors=k).fit(data)
    # Compute the distances to the k nearest neighbors
    distances, _ = knn.kneighbors(data)
    # Calculates anomaly scores based on average distance to its k closest neighbors
    avg_distance = np.mean(distances, axis=1)
    median_avg_distance = np.median(avg_distance)
    std = np.std(avg_distance)
    threshold = median_avg_distance + std
    anomaly_indices = np.where(avg_distance >= threshold)[0]
    contamination = len(anomaly_indices) / len(data)
    return anomaly_indices, contamination

def kmean2d(values, timestamps, n_clusters):
    X = np.column_stack((timestamps, values))
    timeserieskmeans = TimeSeriesKMeans(n_clusters=n_clusters).fit(X)
    distances = timeserieskmeans.transform(X)
    
    cluster_distance = []
    for i in range(len(distances)):
        cluster_label = timeserieskmeans.labels_[i]
        cluster_distance.append(distances[i][cluster_label])
    
    median = np.median(cluster_distance)
    std = np.std(cluster_distance)
    threshold = median + std
    anomaly_indices = np.where(cluster_distance > threshold)[0]
    cluster_labels_array = np.array(timeserieskmeans.labels_)
    contamination = len(anomaly_indices) / len(values)
    return cluster_labels_array, anomaly_indices, contamination

def iso(data, con):
    anomalies = IsolationForest(contamination=con).fit_predict(data)
    anomaly_indices = np.where(anomalies == -1)[0]

    return anomaly_indices

def lof(data, k, con):
    #lof = LocalOutlierFactor(k, contamination='auto')
    lof = LocalOutlierFactor(k, contamination=con)
    anomaly_scores = lof.fit_predict(data)
    anomaly_indices = np.where(anomaly_scores == -1)[0]
    return anomaly_indices

def knnAveraged(values, timestamps):
    anomaly_indices = knn(values, timestamps, 4)
    sum = 0
    for i in range(len(values)):
        if i not in anomaly_indices:
            sum += values[i]
    avg = sum / (len(values) - len(anomaly_indices))
    std_deviation = np.std(np.array(values))
    threshold = avg+std_deviation*1.5
    anomaly_index_high = np.where(np.array(values) > threshold)[0]
    anomaly_index_low = np.where(np.array(values) < threshold)[0]
    return avg, anomaly_index_high, anomaly_index_low, threshold

def calculate_WSS(values, timestamps, graph):
    X = np.array(values).reshape(-1, 1)
    # Initialize lists to store silhouette scores and average silhouette scores
    silhouette_scores = []
    n_clusters_range = range(2, 20)

    # Perform Time Series K-Means clustering for different numbers of clusters
    for n_clusters in n_clusters_range:
        model = TimeSeriesKMeans(n_clusters=n_clusters)
        y_pred = model.fit_predict(X)

        # Calculate the silhouette score for the current number of clusters
        silhouette = silhouette_score(X, y_pred)
        silhouette_scores.append(silhouette)

    # Find the best number of clusters based on the highest silhouette score
    best_n_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 to convert
    print("Best number of cluster: " + str(best_n_clusters) + "\n")
    print("Max score = ", max(silhouette_scores))
    max_ss = max(silhouette_scores)
    # Plot the silhouette scores
    if graph == True:
        plt.figure(figsize=(10, 5))
        plt.plot(n_clusters_range, silhouette_scores, marker='o', linestyle='-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Different Numbers of Clusters')
        plt.grid(True)

        # Show the plots
        plt.show()

    return best_n_clusters, max_ss