import xml.etree.ElementTree as ET
import math
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import base64
import csv
from io import StringIO

def gendata():
    timestamps = [i for i in range(100)]
    values = [i*10 for i in range(100)]
    return timestamps, values

def convert_to_df(timestamps, values):
    data = np.array([timestamps, values])
    print(data)
    dataset = pd.DataFrame({'timestamp': data[0, :], 'value': data[1, :]})
    return dataset

def split_data(timestamps, values):
    # splits data so that 20% is test and 80% is training.

    x_test, x_train = train_test_split(timestamps, test_size=.2, shuffle=False)
    y_test, y_train = train_test_split(values, test_size=.2, shuffle=False)

    return np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)

def parse_xml_data(file, path):
    tree = ET.parse(path + file + '.xml')
    root = tree.getroot()

    # Extract the timestamp and value data from the <database> section
    timestamps = []
    values = []
    for row in root.findall('.//row'):
        timestamp = int(row.find('t').text)
        value = float(row.find('v').text)
        timestamp = datetime.fromtimestamp(timestamp)
        if not math.isnan(value):
            #print(timestamp)
            timestamps.append(timestamp)
            values.append(value)
    
    timestamps_float = [timestamp.timestamp() for timestamp in timestamps]
    data = np.column_stack((timestamps_float, values))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_timestamps, scaled_values = scaled_data[:, 0], scaled_data[:, 1]

    return scaled_timestamps, scaled_values, timestamps, values

def parse_binary_xml_data(filepath):
    # Read the binary file and decode Base64 data
    with open(filepath, 'rb') as f:
        base64_data = f.read()

    # Check if data contains prefix "data:text/xml;base64,"
    prefix = "data:text/xml;base64,"
    if base64_data.startswith(prefix.encode()):
        # Remove the prefix
        base64_data = base64_data[len(prefix):]

    # Decode Base64 data
    xml_data = base64.b64decode(base64_data).decode('utf-8')

    # Parse XML data
    root = ET.fromstring(xml_data)

    # Extract the timestamp and value data from the <database> section
    timestamps = []
    values = []
    for row in root.findall('.//row'):
        timestamp = int(row.find('t').text)
        value = float(row.find('v').text)
        timestamp = datetime.fromtimestamp(timestamp)
        if not math.isnan(value):
            timestamps.append(timestamp)
            values.append(value)

    timestamps_float = [timestamp.timestamp() for timestamp in timestamps]
    data = np.column_stack((timestamps_float, values))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_timestamps, scaled_values = scaled_data[:, 0], scaled_data[:, 1]

    return scaled_timestamps, scaled_values, timestamps, values

def parse_binary_csv(filepath):
    # Read the binary file and decode Base64 data
    with open(filepath, 'rb') as f:
        base64_data = f.read()

    # Check if data contains prefix "data:text/xml;base64,"
    prefix = "data:text/csv;base64,"
    if base64_data.startswith(prefix.encode()):
        # Remove the prefix
        base64_data = base64_data[len(prefix):]

    # Decode Base64 data
    csv_data = base64.b64decode(base64_data).decode('utf-8')

    # Parse CSV data
    timestamps = []
    values = []
    csv_reader = csv.reader(StringIO(csv_data))
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        timestamps.append(timestamp)
        values.append(float(row[1]))

    timestamps_float = [timestamp.timestamp() for timestamp in timestamps]
    data = np.column_stack((timestamps_float, values))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_timestamps, scaled_values = scaled_data[:, 0], scaled_data[:, 1]

    return scaled_timestamps, scaled_values, timestamps, values

def format_csv(filepath):
    #Function to convert from grafana CSV format to CSV format accepted by the anomaly detection tool.
    new_rows = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        header_mapping = {header[i]: i for i in range(len(header))}
        for row in reader:
            if len(row) > 0:
                timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                value = row[1].split()[0]  # Extracting the value without units
                new_row = [timestamp, value]
                new_rows.append(new_row)

    # Writing to a new CSV file with the desired format
    with open('converted_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'value'])
        writer.writerows(new_rows)


if __name__ == "__main__":
    # For testing purposes

    #format_csv("/Users/johnhoofttoomey/OHAZ/Nagios_data/Dash/SPF1RXandTX Rates.csv")
    parse_binary_csv("/Users/johnhoofttoomey/OHAZ/Nagios_data/Dash/useruploads/SPF1_PLC_RX.csv")
    #parse_xml_data("bendodot_router", "/Users/johnhoofttoomey/OHAZ/Nagios_data/data/")
    """
    timestamps, values = gendata()
    x_test, y_test, x_train, y_train = split_data(timestamps, values)
    print(x_test, y_test)
    print('\n')
    print(x_train, y_train)
    print('\n')
    print('\n')
    test_df = convert_to_df(x_test, y_test)
    train_df = convert_to_df(x_train, y_train)
    print(test_df)
    print('\n')
    print(train_df)
    """