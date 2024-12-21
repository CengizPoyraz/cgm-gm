import pandas as pd
from obspy import read, UTCDateTime

# Load the station picks
phase_picks = pd.read_csv('parsed_phase_picks.csv')
copy_phase_picks = phase_picks.copy() # This copy of phase_picks will be used to drop the rows corresponding to the earthquakes for which we don't have the data

# The reason we drop the first two columns is that they contain the station name and times which we need in order to read the data but we don't want them to be in the final dataset
copy_phase_picks = copy_phase_picks.drop(phase_picks.columns[:2], axis=1)

waveform_list = []

for i in range(len(phase_picks)):

    event = phase_picks.iloc[i,:]
    date_and_time = event['time'].split()
    date = date_and_time[0]

    phase_time = date_and_time[1][:-2]

    begin_time = 0

    event_time_utc = UTCDateTime(event['time'])

    letter = 'N' # Change this to 'E' to get the East-West components
    file_name = event['station'] +'_' + date.replace("/", "") + '_' + f'{phase_time[:2]}00' + '_100_B_KO_' + letter + '.GCF'
    file_path = 'PhaseNet Sample Data' + '\\' + f'{date[-2:]}' + '\\' + f'{phase_time[:2]}' + '\\' + file_name

    # The try statement is used to handle the case where an earthquake is given in phase_picks but doesn't exist in our data
    try:

        st = read(file_path)
            
        #Check the start and end times of the waveform
        start_time = st[0].stats.starttime

        end_time = st[0].stats.endtime

        # Calculate desired slice range
        slice_start = max(event_time_utc - 10, start_time)  
        slice_end = min(event_time_utc + 50, end_time)     

        waveform = st.slice(slice_start, slice_end)

        # Applying filter 
        waveform_filtered = waveform.filter('bandpass', freqmin=2.0, freqmax=15.0)

        waveform_filtered = waveform_filtered[0]
        waveform_list.append(waveform_filtered.data)

    except:
        # We remove the rows for which we don't have the data from the copy of phase picks so that the shape of phase picks and our readings matches
        copy_phase_picks = copy_phase_picks.drop(i)

 # Create and save the final dataset       
waveform_df = pd.DataFrame(waveform_list)
copy_phase_picks.reset_index(drop=True, inplace=True)
training_data = pd.concat([copy_phase_picks, waveform_df], axis=1)
training_data.to_csv(f"training_data_{letter}.csv", index=False)

