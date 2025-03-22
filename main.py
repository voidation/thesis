import argparse
import time
import pandas as pd
from datetime import datetime, timedelta
import struct

# Cyton board library barinflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.board_shim import BoardIds, BrainFlowPresets

# this function might not be required - just shows how to get data using python
# instead of using generated files from recording from GUI
def getDataBrainflow():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    # params.ip_port = args.ip_port
    # params.serial_port = args.serial_port
    # params.mac_address = args.mac_address
    # params.other_info = args.other_info
    # params.serial_number = args.serial_number
    # params.ip_address = args.ip_address
    # params.ip_protocol = args.ip_protocol
    # params.timeout = args.timeout
    # params.file = args.file
    # params.master_board = args.master_board
    params.serial_port = "COM8"

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream ()
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()
    print(data)

def readEEGData():
    '''
    Input:
    Output: EEG_df (pandas.dataFrame)
    Description: Reads the RAW text file that is generated from the EEG data
    stream and outputs a dataframe containing the data.
    '''
    data_file = r'C:\Users\User\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2024-07-23_10-36-09\OpenBCI-RAW-2024-07-23_10-36-56.txt'
    EEG_df = pd.read_csv(data_file, sep=",", skiprows=range(4))
    EEG_df.columns = EEG_df.columns.str.strip()
    EEG_df["Timestamp (Formatted)"] = pd.to_datetime(EEG_df["Timestamp (Formatted)"])
    return EEG_df

def findSAStartTime(participant_id):
    '''
    Input: participant_id (str)
    Output: timestamp (datetime)
    Description: This function looks in the "satest-log.txt" file generated from
    the PEBL satest and finds the timestamp of the latest session of the passed
    in participant_id.
    '''
    start_time_file = r'C:\Users\User\Documents\PEBL2.1\battery\satest\data\satest-log.txt'

    # Define the column names
    column_names = ["Event", "Action", "Time_milli", "Day", "Month", "Date", "Time", "Year", "ID"]

    # Read the file into a pandas DataFrame
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
    
    filtered_df['Adjusted_Timestamp'] = filtered_df.apply(lambda row: row['Timestamp'] - timedelta(milliseconds=row['Time_milli']), axis=1)

    # Find the row with the most recent timestamp
    most_recent_session = filtered_df.loc[filtered_df['Adjusted_Timestamp'].idxmax()]

    return most_recent_session['Adjusted_Timestamp']

# this function needs commenting
def readSATestData(participant_id):
    data_file = fr'C:\Users\User\Documents\PEBL2.1\battery\satest\data\{participant_id}\satest-{participant_id}.csv'
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
    
    with open(data_file, 'w') as file:
        file.writelines('\n'.join(fixed_lines))
    
    file.close()

    col_names = ['participant_id', 'test_block', 'test_type', 'test_number', '--', 
                 'activity_start_time', 'test_start_time', '---',
                 'description', '1', '2', '3', '4', '5', '6', '7']
    
    SA_df = pd.read_csv(data_file, header=None, sep=',', names=col_names, dtype=str)

    # List of columns to remove
    columns_to_remove = ['--', '---']

    # Drop the specified columns
    SA_df = SA_df.drop(columns=columns_to_remove)

    SA_df = SA_df.astype({
        'participant_id': 'str',
        'test_block': 'int',
        'test_type': 'int',
        'test_number': 'int',
        'activity_start_time': 'int',
        'test_start_time': 'int',
        'description': 'str',
    })

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

    return df_level1, df_level2, df_level3

def classifySATestData(SA_df_level1, SA_df_level2, SA_df_level3, level1_th, level3_th):
    """
    Input: SA_df_level1, SA_df_level2, SA_df_level3 (pandas.dataFrame), level1_th, level2_th (float)
    Output: level1_df, level2_df, level3_df (pandas.dataFrame)
    Description: Based off the results of the tests, create data frames with times
    and whether the record was a 'low' SA or 'high' SA one.
    """
    level1_df = SA_df_level1[['activity_start_time', 'test_start_time', 'avg_error']].copy()

    # New column SA where False means low SA and True means high SA
    level1_df['SA'] = True  # Default assignment
    level1_df.loc[level1_df['avg_error'] > level1_th, 'SA'] = False
    #print(level1_df)

    level2_df = SA_df_level2[['activity_start_time', 'test_start_time', 'result']].copy()
    # Group by 'activity_start_time' and 'test_start_time'
    level2_df = level2_df.groupby(['activity_start_time', 'test_start_time'])['result']
    # Set level2_df where 'SA' is True if both results in the pair are 1, otherwise False
    level2_df = level2_df.agg(lambda x: (x == 1).all()).reset_index()
    # Rename the 'result' column to 'SA'
    level2_df = level2_df.rename(columns={'result': 'SA'})
    # print(level2_df)

    level3_df = SA_df_level3[['activity_start_time', 'test_start_time', 'angle_diff']].copy()
    # New column SA where False means low SA and True means high SA
    level3_df['SA'] = True  # Default assignment
    level3_df.loc[level3_df['angle_diff'] > level3_th, 'SA'] = False
    
    return level1_df, level2_df, level3_df

def readfNIRSData():
    data_file = r"C:\Users\User\Desktop\Thesis\NIRSDataCollection\DevData\p300hdh_10062024_es_a_1.oxy4"
    with open(data_file, "rb") as f:
        data = f.read()
    f.close()
    with open(r"C:\Users\User\Desktop\Thesis\NIRSDataCollection\DevData\data_in_binary.txt", "w") as file:
        file.write(str(data))
    file.close()

#Figure this out
def labelEEGData(EEG_df, level1_df, level2_df, level3_df):
    print(EEG_df)
    print(level1_df)

    # List to store the aggregated EEG samples and their labels
    aggregated_samples = []
    labels = []

    # Iterate through each row in the DataFrame
    for index, row in level1_df.iterrows():
        start_time = row['activity_start_time']
        end_time = row['test_start_time']
        label = row['SA']
        
        # Extract EEG data between start_time and end_time
        eeg_segment = EEG_df[(EEG_df['Timestamp (Formatted)'] >= start_time) & (EEG_df['Timestamp (Formatted)'] <= end_time)]
        
        # Aggregate the EEG data into a single sample
        # Example: Concatenate the EEG signal values into a single array
        aggregated_sample = eeg_segment['EEG_signal'].values.flatten()
        
        # Store the aggregated sample and its label
        aggregated_samples.append(aggregated_sample)
        labels.append(label)

    # Create a new DataFrame or use a list of tuples for further processing
    # For a DataFrame:
    aggregated_df = pd.DataFrame({'EEG_sample': aggregated_samples, 'Label': labels})

    print(aggregated_df)


def main():
    participant_id = input("What is the participant id?")
    EEG_df = readEEGData()
    SA_df_level1, SA_df_level2, SA_df_level3 = readSATestData(participant_id)
    level1_df, level2_df, level3_df = classifySATestData(
        SA_df_level1, SA_df_level2, SA_df_level3, 0.49, 37.43)
    labelEEGData(EEG_df, level1_df, level2_df, level3_df)
    #readfNIRSData()



if __name__ == "__main__":
    main()