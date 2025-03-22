# Import necessary libraries
import pickle
import re
import pandas as pd
from datetime import datetime, timedelta
import os
import io
import numpy as np
import scipy.signal as signal
from scipy.signal import resample, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom model imports
from MultiInputCNN import MultiInputCNN
from MLP import MLPModel
from Transformer import TransformerModel
import XGBoost
import RandomForest
import KNN

### EEG DATA FUNCTIONS ###

def readEEGData(data_file):
    '''
    Reads raw EEG data from a text file into a pandas DataFrame.
    
    Parameters:
        data_file (str): Path to the EEG data file.
        
    Returns:
        df (pandas.DataFrame): DataFrame containing EEG data with timestamps.
    '''
    # Load data
    df = pd.read_csv(data_file, sep=",", skiprows=range(4))
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Convert timestamps to datetime objects
    df["Timestamp (Formatted)"] = pd.to_datetime(df["Timestamp (Formatted)"])
    return df

def readAllEEGData(path_to_data_folder):
    '''
    Reads and concatenates all EEG data files in a specified folder into a single DataFrame.
    
    Parameters:
        path_to_data_folder (str): Path to the folder containing EEG data files.
        
    Returns:
        EEG_df (pandas.DataFrame): DataFrame containing concatenated EEG data.
    '''
    dataframes = []
    for subdir, dirs, files in os.walk(path_to_data_folder):
        for file in files:
            filename = fr"{subdir + os.sep + file}"
            # Append each EEG file data to the list
            dataframes.append(readEEGData(filename))
    
    # Concatenate all EEG data into one DataFrame, sorted by Timestamp
    EEG_df = pd.concat(dataframes, ignore_index=True)
    EEG_df = EEG_df.sort_values(by="Timestamp (Formatted)")
    EEG_df = EEG_df.reset_index(drop=True)

    return EEG_df

def EEGData(path_to_data_folder):
    '''
    Reads, preprocesses, and filters EEG data for analysis.
    
    Parameters:
        path_to_data_folder (str): Path to the folder containing EEG data files.
        
    Returns:
        EEG_df (pandas.DataFrame): DataFrame containing preprocessed EEG data.
    '''

    # Band Pass Filter function
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    # Notch filter function to remove powerline noise
    def notch_filter(data, freq, fs, quality_factor=30):
        nyquist = 0.5 * fs
        freq = freq / nyquist
        b, a = butter(2, [freq - 0.01, freq + 0.01], btype='bandstop')
        return filtfilt(b, a, data, axis=0)
    
    # Function to replace outliers with median values
    def replace_outliers_with_median(data, threshold=4):
        # Calculate mean and standard deviation along each column
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)

        # Calculate lower and upper bounds for outliers
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev

        # Identify outliers (values outside the lower and upper bounds)
        outliers = (data < lower_bound) | (data > upper_bound)

        # Calculate the median for each column
        median = np.median(data, axis=0)

        # Replace outliers with the median of the respective column
        for col in range(data.shape[1]):  # Iterate through each column
            data[outliers[:, col], col] = median[col]

        return data
    
    # Load and pre-process EEG data
    EEG_df = readAllEEGData(path_to_data_folder)
    # Keep only the useful columns
    EEG_df = EEG_df.drop(
        columns=EEG_df.columns.difference([
            "EXG Channel 0",
            "EXG Channel 1",
            "EXG Channel 2",
            "EXG Channel 3",
            "EXG Channel 4",
            "EXG Channel 5",
            "EXG Channel 6",
            "EXG Channel 7",
            "Timestamp (Formatted)"
            ]), axis=1
    )

    # plt.figure(figsize=(10, 5))
    # plt.plot(EEG_df["Timestamp (Formatted)"].to_list(), EEG_df["EXG Channel 2"].to_list())
    # plt.title("EEG before pre-processing")
    # plt.xlabel("Time")
    # plt.ylabel("EXG Channel 2 Signal")
    # plt.show()

    # Extract EEG signals
    eeg_signals = EEG_df.iloc[:, :-1].values
    
    # Parameters for filtering
    fs = 250  # Sampling frequency (Hz)
    lowcut = 0.5  # Low cutoff frequency
    highcut = 45  # High cutoff frequency

    # Replace the EEG signals in the DataFrame
    EEG_df.iloc[:, :-1] = eeg_signals

    # Apply notch filter to remove power line interference at 50Hz
    filtered_eeg = notch_filter(eeg_signals, freq=50, fs=fs)
    
    # Apply bandpass filter (e.g., 0.5 Hz to 50 Hz) - this removes DC component as well
    filtered_eeg = bandpass_filter(filtered_eeg, lowcut, highcut, fs)

    # Remove outliers
    filtered_eeg = replace_outliers_with_median(filtered_eeg, threshold=4)

    # Replace the EEG signals in the DataFrame
    EEG_df.iloc[:, :-1] = filtered_eeg

    # plt.figure(figsize=(10, 5))
    # plt.plot(EEG_df["Timestamp (Formatted)"].to_list(), EEG_df["EXG Channel 2"].to_list())
    # plt.title("EEG after pre-processing")
    # plt.xlabel("Time")
    # plt.ylabel("EXG Channel 2 Signal")
    # plt.show()

    return EEG_df

### SA DATA FUNCTIONS ###

def findSAStartTime(participant_id):
    '''
    Identifies and returns the start time of a situational awareness test session for a given participant.
    
    Parameters:
        participant_id (str): The ID of the participant.
        
    Returns:
        datetime: The start time of the test session.
    '''
    start_time_file = r'./SA/data/satest-log.txt'

    # Define the column names
    column_names = ["Event", "Action", "Time_milli", "Day", "Month", "Date", "Time", "Year", "ID"]

    # Read the log file into a pandas DataFrame
    df = pd.read_csv(start_time_file, sep='\s+', header=None, names=column_names)

    # Filter the DataFrame for the specific participant_id
    filtered_df = df[df['ID'] == str(participant_id)].copy()

    if filtered_df.empty:
        print(f"No data found for participant_id {participant_id}")
        return None

    # Combine relevant columns to create a full timestamp
    filtered_df['Timestamp'] = filtered_df.apply(lambda row: datetime.strptime(
        f"{row['Day']} {row['Month']} {row['Date']} {row['Time']} {row['Year']}",
        '%a %b %d %H:%M:%S %Y'), axis=1)
    
    # Adjust timestamp by subtracting milliseconds
    filtered_df['Adjusted_Timestamp'] = filtered_df.apply(lambda row: row['Timestamp'] - timedelta(milliseconds=row['Time_milli']), axis=1)

    # Find the row with the most recent timestamp
    most_recent_session = filtered_df.loc[filtered_df['Adjusted_Timestamp'].idxmax()]

    return most_recent_session['Adjusted_Timestamp']

def readSATestData(participant_id):
    """
    Reads and processes situational awareness (SA) test data for a given participant,
    returning separate DataFrames for each SA test level.

    Parameters:
        participant_id (str): The ID of the participant whose data is being read.

    Returns:
        df_level1, df_level2, df_level3 (pandas.DataFrame): Separate DataFrames 
        for each test level, containing relevant columns and formatted data.
    """
    # Define the file path for the participant's SA test data
    data_file = fr'./SA/data/{participant_id}/satest-{participant_id}.csv'
    start_time = findSAStartTime(participant_id)

    # Because level3 tests have more columns, we need to add 2 blank columns
    # to all the other rows in the file before reading it as a dataframe
    with open(data_file, 'r') as file:
        lines = file.readlines()
        num_cols = 16

        fixed_lines = []
        for line in lines:
            # Split the line into parts
            parts = line.split(',')
            # Remove trailing \n characters from each part
            parts = [part.rstrip() for part in parts]
            # Check if the number of cols is correct
            if len(parts) < num_cols:
                # Add blank parts to get to the correct number of cols
                parts.extend([''] * (num_cols - len(parts)))
            # Add commas to the end of each part
            fixed_lines.append(','.join(parts))
        
    file.close()
    
    # Overwrite the file with lines that now have consistent column counts
    with open(data_file, 'w') as file:
        file.writelines('\n'.join(fixed_lines))
    
    file.close()

    # Define the column names for the DataFrame
    col_names = ['participant_id', 'test_block', 'test_type', 'test_number', '--', 
                 'activity_start_time', 'test_start_time', '---',
                 'description', '1', '2', '3', '4', '5', '6', '7']
    
    # Read the adjusted file into a DataFrame with specified column names and as strings
    SA_df = pd.read_csv(data_file, header=None, sep=',', names=col_names, dtype=str)

    # List of columns to remove
    columns_to_remove = ['--', '---']

    # Drop the specified columns
    SA_df = SA_df.drop(columns=columns_to_remove)

    # Convert specific columns to appropriate data types for analysis
    SA_df = SA_df.astype({
        'participant_id': 'str',
        'test_block': 'int',
        'test_type': 'int',
        'test_number': 'int',
        'activity_start_time': 'int',
        'test_start_time': 'int',
        'description': 'str',
    })

    # Adjust timestamps to account for the start of the test session
    SA_df['activity_start_time'] = SA_df.apply(lambda row: start_time + timedelta(milliseconds=row['activity_start_time']), axis=1)
    SA_df['test_start_time'] = SA_df.apply(lambda row: start_time + timedelta(milliseconds=row['test_start_time']), axis=1)

    df_level1 = SA_df[SA_df['test_type'] == 1]
    # List of columns to remove
    columns_to_remove = ['6', '7']
    # Drop the specified columns
    df_level1 = df_level1.drop(columns=columns_to_remove)
    df_level1 = df_level1.rename(columns={
        '1': 'click_number',
        '2': 'resp.x',
        '3': 'resp.y',
        '4': 'click_time',
        '5': 'avg_error'
        })
    
    # Convert multiple columns to their respective types
    df_level1 = df_level1.astype({
        'click_number': 'float',
        'resp.x': 'float',
        'resp.y': 'float',
        'click_time': 'int',
        'avg_error': 'float'
    })

    df_level1 = df_level1.dropna(subset=['avg_error'])
    df_level1 = df_level1.sort_values(by=['test_start_time', 'click_time'])

    df_level2 = SA_df[SA_df['test_type'] == 3]
    # List of columns to remove
    columns_to_remove = ['6', '7']
    # Drop the specified columns
    df_level2 = df_level2.drop(columns=columns_to_remove)
    df_level2 = df_level2.rename(columns={
        '1': 'click_number',
        '2': 'target_id',
        '3': 'target.x',
        '4': 'click_time',
        '5': 'result'
        })
    
    # Convert multiple columns to their respective types
    df_level2 = df_level2.astype({
        'click_number': 'float',
        'target_id': 'int',
        'target.x': 'int',
        'click_time': 'int',
        'result': 'float'
    })

    df_level2 = df_level2.sort_values(by=['test_start_time', 'click_time'])

    df_level3 = SA_df[SA_df['test_type'] == 5]
    df_level3 = df_level3.rename(columns={
        '1': 'target_id',
        '2': 'target_1.x',
        '3': 'target_1.y',
        '4': 'radius',
        '5': 'angle',
        '6': 'click_time',
        '7': 'angle_diff'
        })
    
    df_level3 = df_level3.astype({
        'target_id': 'int',
        'target_1.x': 'int',
        'target_1.y': 'int',
        'radius': 'float',
        'angle': 'int',
        'click_time': 'int',
        'angle_diff': 'float'
    })

    df_level3 = df_level3.sort_values(by=['test_start_time', 'click_time'])

    # Return DataFrames for each level test, each containing relevant columns and data types
    return df_level1, df_level2, df_level3

def classifySATestData(SA_df_level1, SA_df_level2, SA_df_level3, level1_th, level3_th):
    """
    Classifies situational awareness (SA) levels based on thresholds for test levels 1 and 3,
    creating labeled data frames that indicate 'low' or 'high' SA for each test record.
    
    Parameters:
        SA_df_level1, SA_df_level2, SA_df_level3 (pandas.DataFrame): DataFrames for each SA test level.
        level1_th (float): Threshold for 'avg_error' in level 1 to classify 'low' vs. 'high' SA.
        level3_th (float): Threshold for 'angle_diff' in level 3 to classify 'low' vs. 'high' SA.
        
    Returns:
        level1_df, level2_df, level3_df (pandas.DataFrame): DataFrames with SA classification for each level.
    """
    # --- Level 1 Classification ---
    # Extract necessary columns from level 1 DataFrame
    level1_df = SA_df_level1[['activity_start_time', 'test_start_time', 'avg_error']].copy()

    # New column SA where False means low SA and True means high SA
    level1_df['SA'] = True  # Default assignment
    level1_df.loc[level1_df['avg_error'] > level1_th, 'SA'] = False
    #print(level1_df)

    # --- Level 2 Classification ---
    level2_df = SA_df_level2[['activity_start_time', 'test_start_time', 'result']].copy()
    # Group by 'activity_start_time' and 'test_start_time'
    level2_df = level2_df.groupby(['activity_start_time', 'test_start_time'])['result']
    # Set level2_df where 'SA' is True if both results in the pair are 1, otherwise False
    level2_df = level2_df.agg(lambda x: (x == 1).all()).reset_index()
    # Rename the 'result' column to 'SA'
    level2_df = level2_df.rename(columns={'result': 'SA'})
    # print(level2_df)

    # --- Level 3 Classification ---
    level3_df = SA_df_level3[['activity_start_time', 'test_start_time', 'angle_diff']].copy()
    # New column SA where False means low SA and True means high SA
    level3_df['SA'] = True  # Default assignment
    level3_df.loc[level3_df['angle_diff'] > level3_th, 'SA'] = False
    
    return level1_df, level2_df, level3_df

def SAData(path_to_data_folder, combine=False):
    """
    Reads and processes Situational Awareness (SA) test data files from multiple participants, 
    sorts and classifies SA levels, and optionally combines the data from all levels.

    Parameters:
        path_to_data_folder (str): Path to the folder containing participant SA data files.
        combine (bool): If True, combines classified data from all levels into a single DataFrame.
                        Default is False, which returns separate DataFrames for each SA level.

    Returns:
        If combine=True:
            SA_df (pandas.DataFrame): Combined and sorted DataFrame containing classified data from all levels.
        If combine=False:
            SA_df_level1, SA_df_level2, SA_df_level3 (pandas.DataFrame): Separate DataFrames with 
            classified data for each SA test level.
    """

    # Initialize lists to store DataFrames for each level across participants
    dfs1 = []
    dfs2 = []
    dfs3 = []

    # Walk through the directory structure to find each participant's data
    for subdir, dirs, files in os.walk(path_to_data_folder):
        for dir in dirs:
            # Directory name assumed to be participant ID
            participant_id = dir
            # Read SA test data for each participant and append to corresponding level lists
            df1, df2, df3 = readSATestData(participant_id)
            dfs1.append(df1)
            dfs2.append(df2)
            dfs3.append(df3)
    
    # Concatenate all DataFrames for each level from all participants
    df_level1 = pd.concat(dfs1, ignore_index=True)
    df_level2 = pd.concat(dfs2, ignore_index=True)
    df_level3 = pd.concat(dfs3, ignore_index=True)
    
     # Sort each DataFrame by 'activity_start_time' to ensure chronological order
    df_level1 = df_level1.sort_values(by="activity_start_time")
    df_level1 = df_level1.reset_index(drop=True)

    df_level2 = df_level2.sort_values(by="activity_start_time")
    df_level2 = df_level2.reset_index(drop=True)

    df_level3 = df_level3.sort_values(by="activity_start_time")
    df_level3 = df_level3.reset_index(drop=True)

    # Classify data in each level using predefined thresholds for high and low SA
    SA_df_level1, SA_df_level2, SA_df_level3 = classifySATestData(df_level1, df_level2, df_level3, 0.49, 37.43)

    # If 'combine' is set to True, merge classified DataFrames for all levels into a single DataFrame
    if combine == True:
        SA_df = pd.concat([SA_df_level1, SA_df_level2, SA_df_level3], ignore_index=True)
        SA_df = SA_df.sort_values(by="activity_start_time")
        SA_df = SA_df.reset_index(drop=True)

        return SA_df

    # Return separate DataFrames for each level if not combined
    return SA_df_level1, SA_df_level2, SA_df_level3

### NIRS DATA FUNCTIONS ###

def readNIRSData(data_file):
    """
    Reads and processes a single fNIRS data file to extract relevant data columns 
    and timestamps based on the data file's internal structure and format.

    Parameters:
        data_file (str): The path to the fNIRS data file.

    Returns:
        df (pd.DataFrame): DataFrame containing processed fNIRS data with columns 
                           for O2Hb and HHb concentrations and a timestamp column.
    """
    # Open the file and read it
    with open(data_file, 'r') as file:
        lines = file.readlines()

    # Find the start of the data section
    start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Start of measurement"):
            start_time = datetime.strptime(line.split("\t")[1].strip(), "%Y-%m-%d %H:%M:%S.%f")
        if ''.join(line.split()).startswith("12345"):  # Look for the header line of the data
            start_line = i + 1  # Start reading data after this line
            break

    # Read the data section
    data_lines = lines[start_line:]
    
    # Specify column names as per the 'Legend' section provided
    columns = [
        "Sample Number", "Rx1-Tx1 O2Hb", "Rx1-Tx1 HHb", "Rx1-Tx2 O2Hb", "Rx1-Tx2 HHb",
        "Rx1-Tx3 O2Hb", "Rx1-Tx3 HHb", "Rx1-Tx4 O2Hb", "Rx1-Tx4 HHb",
        "Rx2-Tx5 O2Hb", "Rx2-Tx5 HHb", "Rx2-Tx6 O2Hb", "Rx2-Tx6 HHb",
        "Rx2-Tx7 O2Hb", "Rx2-Tx7 HHb", "Rx2-Tx8 O2Hb", "Rx2-Tx8 HHb",
        "Event"
    ]

    # Read data into a DataFrame using space as delimiter and skip initial spaces
    df = pd.read_csv(
        io.StringIO(''.join(data_lines)),
        sep='\s+',
        skipinitialspace=True,
        names=columns,
    )

    # drop columns
    df = df.drop(columns=["Sample Number", "Event"])

    time_interval = timedelta(seconds=1/50) # 50Hz frequency

    # Generate timestamps based on the start_time and interval
    df["Timestamp"] = [start_time + i * time_interval for i in range(len(df))]

    return df

def readAllNIRSData(path_to_data_folder):
    """
    Reads and combines all fNIRS data files within a specified directory, consolidating 
    them into a single DataFrame. 

    Parameters:
        path_to_data_folder (str): Directory path containing multiple fNIRS data files.

    Returns:
        NIRS_df (pd.DataFrame): DataFrame with combined and sorted fNIRS data 
                                from all files, ordered by timestamp.
    """
    dataframes = []
    # Traverse directory to locate and read each individual file
    for subdir, dirs, files in os.walk(path_to_data_folder):
        for file in files:
            filename = fr"{subdir + os.sep + file}"
            # Read individual data files and append to list
            dataframes.append(readNIRSData(filename))
    
    # Concatenate all dataframes and sort by timestamp to ensure chronological order
    NIRS_df = pd.concat(dataframes, ignore_index=True)
    NIRS_df = NIRS_df.sort_values(by="Timestamp")
    NIRS_df = NIRS_df.reset_index(drop=True)

    return NIRS_df

def NIRSData(path_to_data_folder):
    """
    Main function to preprocess and filter all fNIRS data. This function reads all files, 
    applies filtering to remove high-frequency noise, and addresses data outliers.

    Args:
        path_to_data_folder (str): Directory path containing multiple fNIRS data files.

    Returns:
        NIRS_df (pd.DataFrame): Preprocessed fNIRS data with filtered signals and 
                                adjusted outliers.
    """

    # Apply low-pass filter to remove high-frequency noise (e.g., >0.5Hz)
    def lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        cutoff = cutoff / nyquist
        b, a = butter(order, cutoff, btype='low')
        return filtfilt(b, a, data, axis=0)
    
    # Fixing outliers
    def replace_outliers_with_median(data, threshold=4):
        # Calculate mean and standard deviation along each column
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)

        # Calculate lower and upper bounds for outliers
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev

        # Identify outliers (values outside the lower and upper bounds)
        outliers = (data < lower_bound) | (data > upper_bound)

        # Calculate the median for each column
        median = np.median(data, axis=0)

        # Replace outliers with the median of the respective column
        for col in range(data.shape[1]):  # Iterate through each column
            data[outliers[:, col], col] = median[col]

        return data
    
    # Load and consolidate all NIRS data from the specified folder
    NIRS_df = readAllNIRSData(path_to_data_folder)

    # plt.figure(figsize=(10,5))
    # plt.plot(NIRS_df["Timestamp"].to_list(), NIRS_df["Rx1-Tx1 O2Hb"].to_list(), label='O₂Hb')
    # plt.plot(NIRS_df["Timestamp"].to_list(), NIRS_df["Rx1-Tx1 HHb"].to_list(), label='HHb')
    # plt.title("Raw fNIRS before pre-processing")
    # plt.xlabel("Time")
    # plt.ylabel("Concentration")
    # plt.legend()
    # plt.show()

    # Extract NIRS signals (assuming NIRS_df has 'HbO' and 'HbR' columns)
    nirs_signals = NIRS_df.iloc[:, :-1].values

    # Parameters for filtering
    fs = 50  # Sampling frequency (Hz)
    cutoff = 0.7
    
    # Apply bandpass filter to remove high-frequency and low-frequency noise
    filtered_nirs = lowpass_filter(nirs_signals, cutoff, fs)

    # Remove outliers
    filtered_nirs = replace_outliers_with_median(filtered_nirs)

    # Replace the NIRS signals in the DataFrame
    NIRS_df.iloc[:, :-1] = filtered_nirs

    # plt.figure(figsize=(10,5))
    # plt.plot(NIRS_df["Timestamp"].to_list(), NIRS_df["Rx1-Tx1 O2Hb"].to_list(), label='O₂Hb')
    # plt.plot(NIRS_df["Timestamp"].to_list(), NIRS_df["Rx1-Tx1 HHb"].to_list(), label='HHb')
    # plt.title("fNIRS after pre-processing")
    # plt.xlabel("Time")
    # plt.ylabel("Concentration")
    # plt.legend()
    # plt.show()

    return NIRS_df

### FUNCTIONS FOR OBTAINING LABELLED SAMPLES AND SAVING THEM ###

def getLabelledSamples(EEG_df, NIRS_df, SA_df):
    """
    Aligns and extracts labeled EEG and NIRS data samples based on specified time ranges 
    for situational awareness (SA) testing.

    Parameters:
        EEG_df (pd.DataFrame): DataFrame containing EEG data with timestamps.
        NIRS_df (pd.DataFrame): DataFrame containing NIRS data with timestamps.
        SA_df (pd.DataFrame): DataFrame containing SA test data with activity start, test start 
                              timestamps, and SA labels.

    Returns:
        labelled_eeg_samples (list): List of EEG data samples aligned with the SA test intervals.
        labelled_nirs_samples (list): List of NIRS data samples aligned with the SA test intervals.
        labels (list): List of SA labels corresponding to each aligned data sample (1 for HIGH SA, 
                       0 for LOW SA).
    """

    print("Getting signal samples and lining up labels...")
    # Extract EEG channels (columns 'EXG Channel 0' to 'EXG Channel 7') from the EEG dataframe
    eeg_channels = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
                'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7']
    
    # Extract NIRS channels and timestamps from NIRS_df
    nirs_channels = [
        'Rx1-Tx1 O2Hb',
        'Rx1-Tx1 HHb',
        'Rx1-Tx2 O2Hb',
        'Rx1-Tx2 HHb',
        'Rx1-Tx3 O2Hb',
        'Rx1-Tx3 HHb',
        'Rx1-Tx4 O2Hb',
        'Rx1-Tx4 HHb',
        'Rx2-Tx5 O2Hb',
        'Rx2-Tx5 HHb',
        'Rx2-Tx6 O2Hb',
        'Rx2-Tx6 HHb',
        'Rx2-Tx7 O2Hb',
        'Rx2-Tx7 HHb',
        'Rx2-Tx8 O2Hb',
        'Rx2-Tx8 HHb'
        ]
    
    # EEG signals (ignoring the 'Timestamp (Formatted)' column for now)
    eeg_signals = EEG_df[eeg_channels].values
    eeg_timestamps = EEG_df["Timestamp (Formatted)"].values
    
    nirs_signals = NIRS_df[nirs_channels].values
    nirs_timestamps = NIRS_df["Timestamp"].values

    # Initialise arrays for storing the extracted samples and labels
    labelled_eeg_samples = []
    labelled_nirs_samples = []
    labels = []

    # Iterate through SA_df rows for activity_start_time and test_start_time
    for idx, row in SA_df.iterrows():
        activity_start_time = row['activity_start_time']
        test_start_time = row['test_start_time']
        label = row['SA']  # Assume there's a 'label' column in SA_df

        # Filter EEG signals within the time range
        eeg_indices = (eeg_timestamps >= activity_start_time) & (eeg_timestamps <= test_start_time)
        eeg_sample = eeg_signals[eeg_indices]

        # Filter NIRS signals within the time range
        nirs_indices = (nirs_timestamps >= activity_start_time) & (nirs_timestamps <= test_start_time)
        nirs_sample = nirs_signals[nirs_indices]

        # Append the samples and corresponding label
        if len(eeg_sample) > 0 and len(nirs_sample) > 0:
            labelled_eeg_samples.append(eeg_sample)
            labelled_nirs_samples.append(nirs_sample)

            # 1 for HIGH SA and 0 for LOW SA
            if label:
                labels.append(1)
            else:
                labels.append(0)

    # Return the labelled samples
    return labelled_eeg_samples, labelled_nirs_samples, labels

def save_data(eeg_data, nirs_data, labels, directory):
    """
    Saves EEG, NIRS, and label data to the specified directory in .pkl format for future access.

    Parameters:
        eeg_data (list): List of extracted EEG data samples.
        nirs_data (list): List of extracted NIRS data samples.
        labels (list): List of SA labels corresponding to each data sample.
        directory (str): Directory path where the data files will be saved.

    Returns:
        None
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save EEG data, NIRS data, and labels in separate pickle files
    with open(os.path.join(directory, 'eeg_data.pkl'), 'wb') as f:
        pickle.dump(eeg_data, f)
    with open(os.path.join(directory, 'nirs_data.pkl'), 'wb') as f:
        pickle.dump(nirs_data, f)
    with open(os.path.join(directory, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

def load_data(directory):
    """
    Loads saved EEG, NIRS, and label data from the specified directory.

    Args:
        directory (str): Directory path containing the saved EEG, NIRS, and labels .pkl files.

    Returns:
        eeg_data (list): List of extracted EEG data samples.
        nirs_data (list): List of extracted NIRS data samples.
        labels (list): List of SA labels corresponding to each data sample.
    """

    # Load EEG data, NIRS data, and labels from separate pickle files
    with open(os.path.join(directory, 'eeg_data.pkl'), 'rb') as f:
        eeg_data = pickle.load(f)
    with open(os.path.join(directory, 'nirs_data.pkl'), 'rb') as f:
        nirs_data = pickle.load(f)
    with open(os.path.join(directory, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    return eeg_data, nirs_data, labels

### FUNCTIONS FOR FEATURE EXTRACTION/SELECTION ###

def extract_psd_features(data_sample, fs=250, method='fft'):
    """
    Extracts Power Spectral Density (PSD) features from EEG data using the specified method.

    Parameters:
        data_sample (numpy array): EEG data sample with shape (channels, time_points).
        fs (int): Sampling frequency of the EEG data (default is 250Hz).
        method (str): Method for computing PSD, either 'fft' or 'periodogram' (default is 'fft').

    Returns:
        features (list): List of band power values for each EEG band across channels.
    """
    # Function to extract features from Power Spectral Density (PSD)
    eeg_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }
    
    features = []
    
    for channel_data in data_sample.T:
        if method == 'fft':
            f, psd = signal.welch(channel_data, fs=fs, nperseg=len(channel_data))
        elif method == 'periodogram':
            f, psd = signal.periodogram(channel_data, fs=fs, window='flat', nfft=len(channel_data))

        # Calculate band power for each EEG band
        for band, (low, high) in eeg_bands.items():
            idx_band = np.logical_and(f >= low, f <= high)
            band_power = np.mean(psd[idx_band])
            features.append(band_power)
    
    return features

def apply_ica(data):
    """
    Applies Independent Component Analysis (ICA) to the data.

    Parameters:
        data (numpy array): Input data with shape (samples, channels).

    Returns:
        transformed_data (numpy array): Data transformed by ICA.
    """
    ica = FastICA(n_components=data.shape[1], max_iter=1000, tol=1.5, random_state=0)
    return ica.fit_transform(data)

def apply_pca(data):
    """
    Applies Principal Component Analysis (PCA) to the data.

    Parameters:
        data (numpy array): Input data with shape (samples, channels).

    Returns:
        transformed_data (numpy array): Data transformed by PCA.
    """
    pca = PCA(n_components=data.shape[1], random_state=0)
    return pca.fit_transform(data)

def extract_eeg_features(eeg_sample):
    """
    Extracts various features from EEG data, including PSD features and ICA/PCA transformations.

    Parameters:
        eeg_sample (numpy array): EEG data sample with shape (channels, time_points).

    Returns:
        features (list): List of extracted features from EEG data.
    """
    features = []

    # Original EEG features using FFT
    features += extract_psd_features(eeg_sample, method='fft')
    # Original EEG features using periodogram
    features += extract_psd_features(eeg_sample, method='periodogram')

    # ICA and PCA
    ica_eeg = apply_ica(eeg_sample)
    pca_eeg = apply_pca(eeg_sample)

    # Features from ICA of EEG using FFT
    features += extract_psd_features(ica_eeg, method='fft')
    # Features from ICA of EEG using periodogram
    features += extract_psd_features(ica_eeg, method='periodogram')

    # Features from PCA of EEG using FFT
    features += extract_psd_features(pca_eeg, method='fft')
    # Features from PCA of EEG using periodogram
    features += extract_psd_features(pca_eeg, method='periodogram')

    return features

def extract_nirs_features(nirs_sample):
    """
    Extracts features from NIRS data, including ICA transformations.

    Parameters:
        nirs_sample (numpy array): NIRS data sample with shape (channels, time_points).

    Returns:
        features (list): List of extracted features from NIRS data.
    """
    features = []
    
    # Original NIRS features using statistical measures
    features += extract_nirs_stat_features(nirs_sample)

    # ICA of NIRS
    ica_nirs = apply_ica(nirs_sample)

    # Features from ICA of NIRS (calculate necessary features)
    features += extract_nirs_stat_features(np.array(ica_nirs))

    return features

def extract_eeg_stat_features(eeg_sample):
    """
    Extracts statistical features from EEG data, including mean, standard deviation, skew, kurtosis, and entropy.

    Parameters:
        eeg_sample (numpy array): EEG data sample with shape (channels, time_points).

    Returns:
        features (list): List of statistical features for each EEG channel.
    """
    # Function to extract statistical features from EEG signals

    features = []
    # For each channel
    for channel_data in eeg_sample.T:
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            skew(channel_data),
            kurtosis(channel_data),
            entropy(np.abs(channel_data)),
        ])
    
    return features

def extract_nirs_stat_features(nirs_sample):
    """
    Extracts statistical features from NIRS data, including mean, variance, and max values.

    Parameters:
        nirs_sample (numpy array): NIRS data sample with shape (channels, time_points).

    Returns:
        features (list): List of statistical features for each NIRS channel.
    """
    # NIRS feature extraction

    features = []
    # Assuming O2Hb and HHb are paired
    for channel_idx in range(nirs_sample.shape[1] // 2):
        o2hb_data = nirs_sample[:, 2*channel_idx]   # O₂Hb
        hhb_data = nirs_sample[:, 2*channel_idx+1]  # HHb

        # Statistical features
        features.extend([
            np.mean(o2hb_data),
            np.mean(hhb_data),
            np.var(o2hb_data),
            np.var(hhb_data),
            # Difference between O₂Hb and HHb
            np.mean(o2hb_data - hhb_data), 
            # Slope of the difference
            np.mean(np.gradient(o2hb_data - hhb_data)),
            np.max(o2hb_data),
            np.max(hhb_data)
        ])
    return features

def feature_extraction(eeg_samples, nirs_samples):
    """
    Extracts combined EEG and NIRS features for each sample.

    Parameters:
        eeg_samples (list): List of EEG data samples.
        nirs_samples (list): List of NIRS data samples.

    Returns:
        all_features (numpy array): Array of combined EEG and NIRS features.
    """
    # Full feature extraction function
    print("Extracting Features...")
    all_features = []
    # For each sample
    for eeg_sample, nirs_sample in tqdm(zip(eeg_samples, nirs_samples)):
        # Normalise EEG sample
        eeg_sample_normalised = StandardScaler().fit_transform(eeg_sample)
        
        # Normalise NIRS sample
        nirs_sample_normalised = StandardScaler().fit_transform(nirs_sample)

        # Extract features from EEG
        eeg_features = extract_eeg_features(np.array(eeg_sample_normalised)) + extract_eeg_stat_features(np.array(eeg_sample_normalised))
        
        # Extract features from NIRS
        nirs_features = extract_nirs_features(np.array(nirs_sample_normalised))

        # Combine EEG and NIRS features into a single feature vector
        combined_features = eeg_features + nirs_features
        
        all_features.append(combined_features)
    
    return np.array(all_features)

def generate_feature_names(eeg_channels=8, nirs_channels=8):
    """
    Generates descriptive names for each feature extracted from EEG and NIRS data.

    Parameters:
        eeg_channels (int): Number of EEG channels (default is 8).
        nirs_channels (int): Number of NIRS channels (default is 8).

    Returns:
        feature_names (list): List of feature names for each channel and feature type.
    """
    feature_names = []

    # EEG feature names (PSD for FFT and periodogram)
    eeg_methods = ['fft', 'periodogram']
    eeg_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Generate names for EEG band power features
    for method in eeg_methods:
        for band in eeg_bands:
            for ch in range(eeg_channels):
                feature_names.append(f'EEG_Ch{ch}_{method}_{band}_power')
    
    # EEG ICA features
    for method in eeg_methods:
        for band in eeg_bands:
            for ch in range(eeg_channels):
                feature_names.append(f'EEG_Ch{ch}_{method}_ica_{band}_power')

    # EEG PCA features
    for method in eeg_methods:
        for band in eeg_bands:
            for ch in range(eeg_channels):
                feature_names.append(f'EEG_Ch{ch}_{method}_pca_{band}_power')

    # EEG statistical features (mean, std, skew, kurtosis, entropy)
    stat_features = ['mean', 'std', 'skew', 'kurtosis', 'entropy']
    for ch in range(eeg_channels):
        for stat in stat_features:
            feature_names.append(f'EEG_Ch{ch}_{stat}')

    # NIRS feature names
    for ch in range(nirs_channels):  # Assuming paired O2Hb and HHb channels
        # NIRS statistical features for O2Hb and HHb
        feature_names.extend([
            f'NIRS_Ch{ch}_mean_O2Hb',
            f'NIRS_Ch{ch}_mean_HHb',
            f'NIRS_Ch{ch}_var_O2Hb',
            f'NIRS_Ch{ch}_var_HHb',
            f'NIRS_Ch{ch}_mean_diff_O2Hb-HHb',
            f'NIRS_Ch{ch}_slope_diff_O2Hb-HHb',
            f'NIRS_Ch{ch}_max_O2Hb',
            f'NIRS_Ch{ch}_max_HHb'
        ])

        # NIRS ICA features
        feature_names.extend([
            f'NIRS_Ch{ch}_ica_mean_O2Hb',
            f'NIRS_Ch{ch}_ica_mean_HHb',
            f'NIRS_Ch{ch}_ica_var_O2Hb',
            f'NIRS_Ch{ch}_ica_var_HHb',
            f'NIRS_Ch{ch}_ica_mean_diff_O2Hb-HHb',
            f'NIRS_Ch{ch}_ica_slope_diff_O2Hb-HHb',
            f'NIRS_Ch{ch}_ica_max_O2Hb',
            f'NIRS_Ch{ch}_ica_max_HHb'
        ])

    return feature_names

def compute_mutual_information(features, labels, feature_names):
    """
    Computes mutual information scores between each feature and labels.

    Parameters:
        features (numpy array): Array of feature data.
        labels (numpy array): Array of labels.
        feature_names (list): List of feature names.

    Returns:
        mi_df (pd.DataFrame): DataFrame of mutual information scores for each feature.
    """
    # Compute mutual information between each feature and the labels
    mi_scores = mutual_info_classif(features, labels, random_state=42)

    mi_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Mutual Information': mi_scores
    })

    # Sort by highest mutual information
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
    
    return mi_df

def compute_pearson_correlation(features, labels, feature_names):
    """
    Computes Pearson correlation coefficients for each feature against labels.

    Parameters:
        features (numpy array): Array of feature data.
        labels (numpy array): Array of labels.
        feature_names (list): List of feature names.

    Returns:
        correlation_df (pd.DataFrame): DataFrame of Pearson correlations and p-values for each feature.
    """
    correlations = []
    p_values = []
    
    for col in range(features.shape[1]):
        # Compute Pearson correlation for each feature column against the labels
        corr, p_val = pearsonr(features[:, col], labels)
        correlations.append(corr)
        p_values.append(p_val)
    
    correlation_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Pearson Correlation': correlations,
        'P-value': p_values
    })

    # Sort by highest correlation (absolute value)
    correlation_df = correlation_df.reindex(correlation_df['Pearson Correlation'].abs().sort_values(ascending=False).index)
    
    return correlation_df

def perform_rfe(features, labels, n_features_to_select, feature_names):
    """
    Performs Recursive Feature Elimination (RFE) to select the most important features.

    Parameters:
        features (numpy array): Array of feature data.
        labels (numpy array): Array of labels.
        n_features_to_select (int): Number of features to select.
        feature_names (list): List of feature names.

    Returns:
        selected_feature_indices (numpy array): Array of indices of selected features.
        selected_feature_names (list): List of names of selected features.
        rfe_df (pd.DataFrame): DataFrame of features and RFE ranking.
    """
    print("Fitting RFE...")
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(features, labels)

    print("Found best features...")
    # Get indices of selected features
    selected_feature_indices = np.where(rfe.support_)[0]
    
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    
    rfe_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    })
    
    # Return the selected features and the ranking DataFrame
    return selected_feature_indices, selected_feature_names, rfe_df.sort_values(by='Ranking')

### FUNCTIONS FOR MODEL PREP ###

def pepare_for_training_models(features, labels, selected_features = None, smote = True):
    """
    Prepares data for model training by selecting features, splitting data into train and test sets,
    applying SMOTE if specified, and normalizing features.

    Parameters:
        features (numpy array): Array of features.
        labels (numpy array): Array of labels corresponding to the features.
        selected_features (list, optional): List of selected feature indices for feature selection. Default is None.
        smote (bool, optional): If True, applies SMOTETomek to balance classes. Default is True.

    Returns:
        X_train_scaled (numpy array): Scaled training feature data.
        y_train (numpy array): Training labels.
        X_test_scaled (numpy array): Scaled test feature data.
        y_test (numpy array): Test labels.
    """
    print("Preparing data for training...")
    # Select the most important features, if provided
    if selected_features is not None:
        features = features[:, selected_features]

    # Separate data into two classes (class 0 and class 1)
    class_0 = features[labels == 0]
    class_1 = features[labels == 1]
    labels_0 = labels[labels == 0]
    labels_1 = labels[labels == 1]

    print("\n")
    print(f"Shape of class_0: {class_0.shape}, labels_0: {labels_0.shape}")
    print(f"Shape of class_1: {class_1.shape}, labels_1: {labels_1.shape}")

    # Split each class separately into train and test
    class_0_train, class_0_test, labels_0_train, labels_0_test = train_test_split(
        class_0, labels_0, test_size=0.3, random_state=42
    )
    
    class_1_train, class_1_test, labels_1_train, labels_1_test = train_test_split(
        class_1, labels_1, test_size=len(labels_0_test), random_state=42
    )

    print(f"Shape of class_0_train: {class_0_train.shape}, class_0_test: {class_0_test.shape}")
    print(f"Shape of class_1_train: {class_1_train.shape}, class_1_test: {class_1_test.shape}")

    # Combine class 0 and class 1 test sets (for a 50/50 balanced test set)
    X_test = np.concatenate([class_0_test, class_1_test], axis=0)
    y_test = np.concatenate([labels_0_test, labels_1_test], axis=0)

    # Combine the remaining data for training
    X_train = np.concatenate([class_0_train, class_1_train], axis=0)
    y_train = np.concatenate([labels_0_train, labels_1_train], axis=0)

    if smote == True:
        # Apply SMOTETomek to the training set
        smote_tomek = SMOTETomek(random_state=25)
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
        class_counts = np.bincount(y_train)
        print(f"\nThis is the class distribution of training after resampling: {class_counts}")
        class_counts = np.bincount(y_test)
        print(f"This is the class distribution in the test set: {class_counts}")

    # Normalise the feature values (use StandardScaler for normalisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test

def save_data_for_training(X_train, y_train, X_test, y_test, directory):
    """
    Saves training and testing data as pickle files for future use.

    Parameters:
        X_train (numpy array): Training feature data.
        y_train (numpy array): Training labels.
        X_test (numpy array): Test feature data.
        y_test (numpy array): Test labels.
        directory (str): Directory path to save the pickle files.
    """
    print("Saving data for training...")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(os.path.join(directory, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(directory, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(directory, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(directory, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)

def load_data_for_training(directory):
    """
    Loads saved training and testing data from pickle files.

    Parameters:
        directory (str): Directory path containing the pickle files.

    Returns:
        X_train (numpy array): Training feature data.
        y_train (numpy array): Training labels.
        X_test (numpy array): Test feature data.
        y_test (numpy array): Test labels.
    """
    print("Loading data for training...")
    with open(os.path.join(directory, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(directory, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(directory, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(directory, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

def generateLabelledDataset(EEG_df, NIRS_df, SA_df):
    """
    ** NOT IN USE IN MAIN **
    Generates labeled EEG and NIRS data samples aligned by windows of data, and labels them for use in model training.

    Args:
        EEG_df (pd.DataFrame): DataFrame containing EEG data.
        NIRS_df (pd.DataFrame): DataFrame containing NIRS data.
        SA_df (pd.DataFrame): DataFrame containing Situational Awareness (SA) data with labels.

    Returns:
        eeg_windows (numpy array): Array of windowed EEG data.
        nirs_windows (numpy array): Array of windowed and aligned NIRS data.
        labels (numpy array): Array of labels for each data window.
    """

    ## HELPER FUNCTIONS ##
    def window_data(data, timestamps, window_size, stride):
        num_windows = (data.shape[0] - window_size) // stride + 1
        windows = []
        window_times = []

        print("Generating windows...")
        for i in tqdm(range(0, num_windows * stride, stride)):
            # Extract the windowed data
            window = data[i:i + window_size]

            # Get the start and end timestamps for the window
            window_start_time = timestamps[i]
            window_end_time = timestamps[i + window_size - 1]

            # Store window and its corresponding times
            windows.append(window)
            window_times.append((window_start_time, window_end_time))

        return np.array(windows), window_times
    
    def align_nirs_to_eeg(nirs_signals, nirs_timestamps, eeg_window_times, nirs_window_size):
        # Function to downsample and align NIRS data to match the EEG window times
        aligned_nirs_windows = []
        valid_eeg_indices = []
        valid_eeg_window_times = []
        
        print("Aligning NIRS windows to EEG windows...")
        for idx, (window_start_time, window_end_time) in enumerate(tqdm(eeg_window_times)):
            # Find NIRS data within the same time window as the EEG window
            mask = (nirs_timestamps >= window_start_time) & (nirs_timestamps <= window_end_time)
            nirs_window = nirs_signals[mask]

            if len(nirs_window) > 0:
                # If there is NIRS data, resample if necessary
                if len(nirs_window) != nirs_window_size:
                    # If the NIRS window has a different number of samples, resample to match NIRS window size
                    nirs_window = resample(nirs_window, nirs_window_size)

                # Append the aligned NIRS window
                aligned_nirs_windows.append(nirs_window)
                valid_eeg_indices.append(idx)
                valid_eeg_window_times.append((window_start_time, window_end_time))

        return np.array(aligned_nirs_windows), valid_eeg_indices, valid_eeg_window_times

    def label_windows(eeg_window_times, SA_df):
        # Initialize a list to store labels
        labels = []

        print("Labelling each window...")
        # For each window of data
        for window_start_time, window_end_time in tqdm(eeg_window_times):
            # Default label is -1 for windows outside of any SA period
            label = -1

            # Iterate through SA_df rows to check if the window falls in an SA period
            for idx, row in SA_df.iterrows():
                activity_start_time = row['activity_start_time']
                test_start_time = row['test_start_time']
                SA = row['SA']  # True or False

                # If the window falls within a "labellable" period
                if (window_start_time >= activity_start_time) and (window_end_time <= test_start_time):
                    label = 1 if SA else 0
                    break  # Exit loop once a label is found

            labels.append(label)

        return np.array(labels)
    
    # Example window size (time_steps) and channels
    eeg_window_size = 250  # Time steps for EEG
    nirs_window_size = 50 # Time steps for NIRS
    stride = 225 # How much to slide the window (e.g., 10% overlap for EEG)

    # Extract EEG channels (columns 'EXG Channel 0' to 'EXG Channel 7') from the EEG dataframe
    eeg_channels = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
                'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7']
    
    # EEG signals (ignoring the 'Timestamp (Formatted)' column for now)
    eeg_signals = EEG_df[eeg_channels].values
    eeg_timestamps = EEG_df["Timestamp (Formatted)"].values

    # Extract NIRS channels and timestamps from NIRS_df
    nirs_channels = [
        'Rx1-Tx1 O2Hb',
        'Rx1-Tx1 HHb',
        'Rx1-Tx2 O2Hb',
        'Rx1-Tx2 HHb',
        'Rx1-Tx3 O2Hb',
        'Rx1-Tx3 HHb',
        'Rx1-Tx4 O2Hb',
        'Rx1-Tx4 HHb',
        'Rx2-Tx5 O2Hb',
        'Rx2-Tx5 HHb',
        'Rx2-Tx6 O2Hb',
        'Rx2-Tx6 HHb',
        'Rx2-Tx7 O2Hb',
        'Rx2-Tx7 HHb',
        'Rx2-Tx8 O2Hb',
        'Rx2-Tx8 HHb'
        ]
    
    nirs_signals = NIRS_df[nirs_channels].values
    nirs_timestamps = NIRS_df["Timestamp"].values

    # Apply windowing to the EEG signals
    eeg_windows, eeg_window_times = window_data(eeg_signals, eeg_timestamps, eeg_window_size, stride)
    print("EEG windows generated.")

    # Reshape to (num_samples, time_steps, num_channels, 1) for Conv2D input
    eeg_windows = eeg_windows.reshape(eeg_windows.shape[0], eeg_window_size, len(eeg_channels), 1)
    
    # Align and resample the NIRS data to the NIRS window size (50 time steps)
    nirs_windows, valid_eeg_indices, valid_eeg_window_times = align_nirs_to_eeg(nirs_signals, nirs_timestamps, eeg_window_times, nirs_window_size)
    print("NIRS windows generated.")

    # Reshape NIRS windows to (num_samples, time_steps, num_channels, 1) for Conv2D input
    nirs_windows = nirs_windows.reshape(nirs_windows.shape[0], nirs_window_size, len(nirs_channels), 1)

    eeg_windows = eeg_windows[valid_eeg_indices]
    eeg_window_times = valid_eeg_window_times

    # Label the windows based on SA_df
    labels = label_windows(eeg_window_times, SA_df)
    print("Windows labelled.")

    # Filter out windows with label = -1
    valid_indices = labels != -1
    eeg_windows = eeg_windows[valid_indices]
    nirs_windows = nirs_windows[valid_indices]
    labels = labels[valid_indices]

    # Return data suitable for TensorFlow training (EEG, NIRS, and labels)
    return eeg_windows, nirs_windows, labels

def main():
    # Clear terminal output (only works on Windows). Remove this line if not using Windows.
    os.system("cls")

    # Flags to control data generation and model training phases
    generate_data = True
    develop_models = False

    # List of feature counts to experiment with during feature selection (commented options for further testing)
    num_features_list = [10, 20, 40, 80, 160, 200, 300, 408]

    # Toggle between auto feature selection (RFE) and manual selection
    auto_select_features = True

    # Regex pattern for manually selecting certain features (beta, gamma bands, and NIRS ICA features)
    manual_feature_selection_pattern = re.compile(r'(beta|gamma|(?=.*NIRS)(?=.*ica))')

    # Set num_features_list to a list length of 1 if using manual_feature_selection

    ### TRAINING DATA GENERATION ###

    if generate_data == True:
        # Define directory to save/load pre-processed data
        data_dir = "./data"

        # Check if pre-saved data exists; if so, load it
        if os.path.exists(os.path.join(data_dir, 'eeg_data.pkl')):
            print("Loading pre-saved data...")
            eeg_samples, nirs_samples, labels = load_data(data_dir)
        else:
            # Load EEG, SA and NIRS data
            EEG_df = EEGData("./EEG/data")
            print("EEG DataFrame loaded.")
            SA_df = SAData("./SA/data", combine=True)
            print("SA DataFrame loaded.")
            NIRS_df = NIRSData("./NIRS/data")
            print("NIRS DataFrame loaded.")

            # Label samples based on SA levels
            eeg_samples, nirs_samples, labels = getLabelledSamples(EEG_df, NIRS_df, SA_df)

            # Save processed data for future use
            save_data(eeg_samples, nirs_samples, labels, data_dir)
            print("Data saved.")
        
        # Convert labels into numpy array
        labels = np.array(labels)
        # Extract features from pre-processed data
        features = feature_extraction(eeg_samples, nirs_samples)

        print(f"\nFeatures shape: {features.shape}")

        # Generate feature names
        feature_names = generate_feature_names(eeg_channels=8, nirs_channels=8)

        print(f"\nFeature Names length: {len(feature_names)}")
        
        if auto_select_features == True:
            # For every "num_feature" we want to generate dataset
            for num_features in num_features_list:
                print(f"Generating training data for {num_features}...")
                train_dir = f"./data_forTraining_{num_features}features"
                # If path already exists, no need to re-generate
                if os.path.exists(os.path.join(train_dir, 'X_train.pkl')):
                    print("Loading pre-saved model training data...")
                    X_train, y_train, X_test, y_test = load_data_for_training(train_dir)
                else:
                    # Compute Pearson correlation
                    # correlation_df = compute_pearson_correlation(features, labels, feature_names)
                    # print("\nTop 10 features by Pearson correlation:")
                    # print(correlation_df.head(10))

                    # # Compute Mutual Information
                    # mi_df = compute_mutual_information(features, labels, feature_names)
                    # print("\nTop 10 features by Mutual Information:")
                    # print(mi_df.head(10))

                    # Perform RFE to select specific number of features
                    selected_features, _,  rfe_df = perform_rfe(features, labels, n_features_to_select=num_features, feature_names=feature_names)
                    print("\nSelected features by RFE:")
                    print(selected_features)
                    print("\nRFE ranking:")
                    print(rfe_df.head(10))

                    # Prepare data for training with the selected features
                    X_train, y_train, X_test, y_test = pepare_for_training_models(
                    features, labels, selected_features=selected_features, smote=True)

                    # Save the prepared data for future use
                    save_data_for_training(X_train, y_train, X_test, y_test, train_dir)
        else:
             # Manually select features based on the specified pattern
            print("Selecting features manually...")
            selected_features = np.array(
                [i for i, string in enumerate(feature_names) if manual_feature_selection_pattern.search(string)], dtype=np.intp
            )

            print(np.array(feature_names)[selected_features])

            # Prepare data using manually selected features
            X_train, y_train, X_test, y_test = pepare_for_training_models(
                features, labels, selected_features=selected_features, smote=True)

            # Save the prepared data for manual feature selection
            save_data_for_training(X_train, y_train, X_test, y_test, "./data_forTraining_manuallySelectedFeatures")


            print(f"\nTraining data shape: {X_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Testing data shape: {X_test.shape}")
            print(f"Testing labels shape: {y_test.shape}")

    ### MODEL DEVELOPMENT ###

    if develop_models == True:
        # Iterate over each feature count and train models based on pre-saved training data
        if auto_select_features == True:
            for num_features in num_features_list:
                train_dir = f"./data_forTraining_{num_features}features"

                # Check if training data for the feature count exists; if so, proceed to model training
                if os.path.exists(os.path.join(train_dir, 'X_train.pkl')):
                    print("Loading pre-saved model training data...")
                    X_train, y_train, X_test, y_test = load_data_for_training(train_dir)

                    # Train and evaluate KNN model
                    print("\nTraining KNN...\n")

                    model_knn, pca = KNN.train_model(X_train, y_train)
                    KNN.evaluate_model(model_knn, pca, X_test, y_test, output_dir=f"./Models_Performance/KNN/{num_features}features")
                    KNN.save_model(model_knn, "./Models/KNN/", f"knn_model_{num_features}features.pkl")
                    #KNN.visualize_knn_clusters_pca(model_knn, pca, X_train, y_train)

                    # Train and evaluate Boosted Trees model (XGBoost)
                    print("\nTraining Boosted Trees...\n")

                    model_bt = XGBoost.train_model(X_train, y_train)
                    XGBoost.evaluate_model(model_bt, X_test, y_test, output_dir=f"./Models_Performance/XGB/{num_features}features")
                    XGBoost.save_model(model_bt, "./Models/XGB/", f"xgb_model_{num_features}features.pkl")

                    # Train and evaluate Multi-Layer Perceptron (MLP)
                    print("\nTraining MLP...\n")

                    mlp_obj = MLPModel(X_train.shape[1])
                    mlp_obj.train_model(X_train, y_train, f"./Models/MLP/mlp_model_{num_features}features.h5")
                    mlp_obj.evaluate_model(X_test, y_test, output_dir=f"./Models_Performance/MLP/{num_features}features")

                    # Train and evaluate Transformer model
                    print("\nTraining Transformer...\n")

                    # Reshape X_train to (num_samples, 1, num_features)
                    transformer = TransformerModel(X_train.shape[1])
                    transformer.train_model(X_train, y_train, f"./Models/Transformer/transformer_model_{num_features}features.h5")
                    transformer.evaluate_model(X_test, y_test, output_dir=f"./Models_Performance/Transformer/{num_features}features")
                else:
                    print(f"Data for {num_features} does not exist.")
        else:
            # Train models on manually selected features
            train_dir = "./data_forTraining_manuallySelectedFeatures"

            if os.path.exists(os.path.join(train_dir, 'X_train.pkl')):
                print("Loading pre-saved model training data...")
                X_train, y_train, X_test, y_test = load_data_for_training(train_dir)

                print("\nTraining KNN...\n")

                model_knn, pca = KNN.train_model(X_train, y_train)
                KNN.evaluate_model(model_knn, pca, X_test, y_test, output_dir=f"./Models_Performance/KNN/ManuallySelectedfeatures")
                KNN.save_model(model_knn, "./Models/KNN/", f"knn_model_ManuallySelectedfeatures.pkl")
                #KNN.visualize_knn_clusters_pca(model_knn, pca, X_train, y_train)

                print("\nTraining Boosted Trees...\n")

                model_bt = XGBoost.train_model(X_train, y_train)
                XGBoost.evaluate_model(model_bt, X_test, y_test, output_dir=f"./Models_Performance/XGB/ManuallySelectedfeatures")
                XGBoost.save_model(model_bt, "./Models/XGB/", f"xgb_model_ManuallySelectedfeatures.pkl")

                print("\nTraining MLP...\n")

                mlp_obj = MLPModel(X_train.shape[1])
                mlp_obj.train_model(X_train, y_train, f"./Models/MLP/mlp_model_ManuallySelectedfeatures.h5")
                mlp_obj.evaluate_model(X_test, y_test, output_dir=f"./Models_Performance/MLP/ManuallySelectedfeatures")

                print("\nTraining Transformer...\n")

                # Reshape X_train to (num_samples, 1, num_features)
                transformer = TransformerModel(X_train.shape[1])
                transformer.train_model(X_train, y_train, f"./Models/Transformer/transformer_model_ManuallySelectedfeatures.h5")
                transformer.evaluate_model(X_test, y_test, output_dir=f"./Models_Performance/Transformer/ManuallySelectedfeatures")
            else:
                print(f"Data for ManuallySelectedfeatures does not exist.")

if __name__ == "__main__":
    main()