''' 
Data preprocess for generative models
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import scipy.signal
import torch
from geopy import distance
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchaudio
import os
from gms.gms_preprocessor import WaveformPreprocessor

def load_and_preprocess_data(path, data_file_path, idx_file_path, time_spec_converter, train_bs=32, test_bs=32, tcondvar=0, loc='EW'):
    """
    Load and preprocess waveform data with consistent signature
    
    Args:
        path: Base data directory path
        data_file_path: Path to data CSV file
        idx_file_path: Path to index file
        time_spec_converter: Not used, kept for signature compatibility
        train_bs: Training batch size
        test_bs: Testing batch size
        tcondvar: Selection of conditional variables
        loc: Location string
        
    Returns:
        train_set, test_set, all_set, train_loader, test_loader, all_loader, norm_dict, time_serie_len
    """
    print('========================')
    print('Loading data...')
    print('========================\n')

    # Initialize preprocessor
    preprocessor = WaveformPreprocessor(
        window_length=160,  # These could be made parameters if needed
        hop_length=46,
        n_fft=160,
        power=1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Load data file
    if data_file_path is None:
        data_file_path = os.path.join(path, f'Time_Series_Data_v5_{loc}.csv')
    else:
        data_file_path = os.path.join(path, data_file_path)
    if idx_file_path is None:
        idx_file_path = os.path.join(path, f'rand_idx_{loc}.npy')
    else:
        idx_file_path = os.path.join(path, idx_file_path)

    print(f"data file: {data_file_path}, idx file: {idx_file_path}")
    data = pd.read_csv(data_file_path)
    
    # Apply index file if exists
    if os.path.exists(idx_file_path):
        ridx = np.load(idx_file_path)
        if len(ridx) == data.shape[0]:
            data = data.iloc[ridx]

    # Process metadata
    names = ["eid", "src_lat", "src_lon", "sta_lat", "sta_lon", "depth",
             "mag", "Sampling Frequency", "Number of Time Steps", "Delta t",
             "Min Frequency", "Max Frequency", "Number of Frequency Steps", "rup"]

    info = data.iloc[:, :14]
    info.columns = names
    
    # Compute azimuth angles
    angle = compute_angle(info)
    info['angle'] = angle

    # Extract waveforms and remove first 1000 samples
    wfs = data.iloc[:, 14:].to_numpy()[:, 1000:]  # Remove first 1000 samples
    time_serie_len = wfs.shape[-1]
    
    sampling_freq = info['Sampling Frequency'].iloc[0]
    print(f"Waveform shape after trimming: {wfs.shape}")
    print(f"Sampling frequency: {sampling_freq} Hz")

    # Process conditional variables
    cond_var_dict = {
        0: ['mag', 'rup', 'angle'],
        1: ['mag', 'rup', 'angle', 'depth'],
        2: ['mag', 'src_lat', 'src_lon', 'sta_lat', 'sta_lon', 'depth'],
        3: ['mag', 'depth'],
        4: ['mag', 'rup', 'depth']
    }
    
    cond_vars = []
    norm_dict = {}

    # Process and normalize conditional variables
    for cvar in cond_var_dict[tcondvar]:
        cv = info[cvar].to_numpy()
        cv = cv.reshape(cv.shape[0], 1)
        cv_min, cv_max = cv.min(), cv.max()
        norm_dict[cvar] = [cv_min, cv_max]
        cv = min_max_norm(cv, cv_min, cv_max, '[0,1]', 'sub')
        cond_vars.append(cv)

    # Process waveforms in batches
    batch_size = 100
    total_samples = len(wfs)
    all_magnitudes = []
    all_phases = []
    all_norm_dicts = []
    
    print(f"\nProcessing {total_samples} waveforms in batches of {batch_size}")
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch = wfs[i:end_idx]
        
        try:
            print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            magnitudes, phases, batch_norm_dict = preprocessor.batch_process(
                batch,
                sampling_freq=sampling_freq,
                validate=False
            )
            
            if magnitudes is not None and len(magnitudes) > 0:
                all_magnitudes.append(magnitudes)
                all_phases.append(phases)
                all_norm_dicts.append(batch_norm_dict)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue
    
    if not all_magnitudes:
        raise RuntimeError("No waveforms were successfully processed")
    
    # Combine results
    magnitudes = torch.cat(all_magnitudes, dim=0)
    phases = torch.cat(all_phases, dim=0)
    
    # Update norm_dict with waveform normalization
    norm_dict['log_wfs'] = [
        min(d['wf_min'] for d in all_norm_dicts),
        max(d['wf_max'] for d in all_norm_dicts)
    ]

    # Combine conditional variables
    cond_var = np.concatenate(cond_vars, axis=1)
    cond_var = torch.from_numpy(cond_var)
    
    # Convert waveforms to tensor
    wfs = torch.from_numpy(wfs).unsqueeze(1).float()  # Add channel dimension

    # Split into train/test sets
    length = len(magnitudes)
    train_idx = int(length * 0.8)
    
    train_set = [
        magnitudes[:train_idx],
        cond_var[:train_idx],
        phases[:train_idx],
        wfs[:train_idx]
    ]
    
    test_set = [
        magnitudes[train_idx:],
        cond_var[train_idx:],
        phases[train_idx:],
        wfs[train_idx:]
    ]
    
    all_set = [magnitudes, cond_var, phases, wfs]

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(*train_set),
        batch_size=train_bs,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(*test_set),
        batch_size=test_bs,
        shuffle=True
    )
    
    all_loader = DataLoader(
        TensorDataset(*all_set),
        batch_size=length,
        shuffle=False
    )

    return (train_set, test_set, all_set, train_loader, test_loader,
            all_loader, norm_dict, time_serie_len)
 
def load_data(path, data_file_path, idx_file_path, time_spec_converter, train_bs=32, test_bs=32, tcondvar=0, loc='EW'):
    '''
    Arguments
    =========
        data_file_path: file path to the dataset (csv file).
        idx_file_path: file path to the idx formatted file (npy file).
        time_spec_converter: STFT and inverse STFT.
        train_bs: batch size for training
        test_bs: batch size for testing
        tcondvar: selection of conditional variables
        
    Returns
    =======
        datasets, dataloader, norm_dict, and time_series_len
        norm_dict: a dictionary saving all normalization factors
        time_series_len: length of time series
    '''

    if data_file_path is None:
        data_file_path = os.path.join(path, f'Time_Series_Data_v5_{loc}.csv')
    else:
        data_file_path = os.path.join(path, data_file_path)
    if idx_file_path is None:
        idx_file_path = os.path.join(path, f'rand_idx_{loc}.npy')
    else:
        idx_file_path = os.path.join(path, idx_file_path)

    print(f"data file: {data_file_path}, idx file: {idx_file_path}")
    # customized load filtered data
    data = pd.read_csv(data_file_path)
    if os.path.exists(idx_file_path):
        ridx = np.load(idx_file_path)
        if len(ridx) == data.shape[0]:
            data = data.iloc[ridx]

    # assign names for the first 13 columns
    names = ["eid",    #"Event ID"
            "src_lat", #"Source Lat"
            "src_lon", #"Source Lon"
            "sta_lat", #"Station Lat"
            "sta_lon", #"Station Lon"
            "depth",   #"Depth (km)"
            "mag",     #"Magnitude"
            "Sampling Frequency",
            "Number of Time Steps",
            "Delta t",
            "Min Frequency",
            "Max Frequency",
            "Number of Frequency Steps",
            "rup"]     #"rupture distance"

    # define the information for the recorded data
    info = data.iloc[:, :14]
    info.columns = names # the first 14 columns are names for our data
    
    # compute azimuth angles
    angle = compute_angle(info)
    info['angle'] = angle

    # get the time series data
    wfs = data.iloc[:, 14:].to_numpy() # [5108, 7000]

    # cut waveforms to 6000, delete the first 1000 samples
    wfs = wfs[:, np.newaxis, (1000):] # [5108, 1, 6000]
    time_serie_len = wfs.shape[-1]

    # we provide multiple combinations of conditional variables
    cond_var_dict = {0: ['mag', 'rup', 'angle'],
                     1: ['mag', 'rup', 'angle', 'depth'],
                     2: ['mag', 'src_lat', 'src_lon', 'sta_lat', 'sta_lon', 'depth'],
                     3: ['mag', 'depth'],
                     4: ['mag', 'rup', 'depth']}
    cond_vars = []
    norm_dict = {}

    for cvar in cond_var_dict[tcondvar]:
        cv = info[cvar].to_numpy()
        cv = cv.reshape(cv.shape[0], 1)
        cv_min, cv_max = cv.min(), cv.max()
        norm_dict[cvar] = [cv_min, cv_max]
        # normalized by min and max to [0, 1]
        cv = min_max_norm(cv, cv_min, cv_max, '[0,1]', 'sub')
        cond_vars.append(cv)

    # change from time domain to time-frequency domain using STFT
    orig_wfs = torch.from_numpy(wfs).float()
    wfs = time_spec_converter.time_to_spec(orig_wfs.squeeze())
    true_phase, wfs = get_phase_mag(wfs)
    
    # add a threshold of 1e-10
    wfs += 1e-10
    wfs = torch.log10(wfs)
    wfs = wfs.permute(0, 2, 1)

    wfs_min, wfs_max = wfs.min(), wfs.max()
    wfs = min_max_norm(wfs, wfs_min, wfs_max, '[0,1]', 'sub')
    print(f"[load_data] wfs.shape={wfs.shape}")

    norm_dict['log_wfs'] = [wfs_min, wfs_max]

    cond_var = np.concatenate(cond_vars, axis=1) # [5108, len(cond_var_dict[tcondvar])]
    
    cond_var = torch.from_numpy(cond_var)
    length = wfs.shape[0]

    train_set = [wfs[0:int(length * 0.8)], cond_var[0:int(length * 0.8)], true_phase[0:int(length * 0.8)], orig_wfs[0:int(length * 0.8)]]
    test_set = [wfs[int(length * 0.8):], cond_var[int(length * 0.8):], true_phase[int(length * 0.8):], orig_wfs[int(length * 0.8):]]
    all_set = [wfs, cond_var, true_phase, orig_wfs]

    # create dataloaders
    train_loader = DataLoader(TensorDataset(*[train_set[idx] for idx in range(4)]), batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(TensorDataset(*[test_set[idx] for idx in range(4)]), batch_size=int(length * 0.2), shuffle=True)
    all_loader = DataLoader(TensorDataset(*[all_set[idx] for idx in range(4)]), batch_size=length, shuffle=False)

    return train_set, test_set, all_set, train_loader, test_loader, all_loader, norm_dict, time_serie_len


def get_phase_mag(wfs):
    phase = torch.angle(wfs)
    magnitude = torch.abs(wfs)
    return phase, magnitude


def compute_angle(info):

    '''
    Calculate the angle between source centers and station locations
    '''

    # source location: (lat, long)
    src_lat = info['src_lat'].to_numpy() # [#samples,]
    src_lon = info['src_lon'].to_numpy()

    # station location: (lat, long)
    station_lat = info['sta_lat'].to_numpy() 
    station_lon = info['sta_lon'].to_numpy() 

    # calculate the source-site angle
    angle = []
    for i in range(len(src_lat)):
        src_coord = (src_lat[i], src_lon[i])
        station_coord = (station_lat[i], station_lon[i])

        # Calculate the vector from the source center to the station location
        vector = np.array(station_coord) - np.array(src_coord)

        # Calculate the angle between the vectors and the x-axis (east direction)
        angle_rad = np.arctan2(vector[1], vector[0])  # Angle in radians

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        angle.append(angle_deg)
        
    angle = np.array(angle).reshape(len(angle),1)

    return angle


def source_site_distance(info):
    '''compute the source-site distance'''
    # reference: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
    # src_loc: (lat, long)

    # source location: (lat, long)
    src_lat = info['src_lat'].to_numpy() # [#sample,]
    src_lon = info['src_lon'].to_numpy()

    # station location: (lat, long)
    station_lat = info['sta_lat'].to_numpy() 
    station_lon = info['sta_lon'].to_numpy() 

    # calculate the source-site distance
    dist = []
    for i in range(len(src_lat)):
        dist_val = distance.distance((src_lat[i], src_lon[i]), (station_lat[i], station_lon[i])).km
        dist.append(dist_val)

    dist = np.array(dist).reshape(len(dist),1) 

    return dist


def min_max_norm(x, x_min, x_max, range='[0,1]', mode='add'):

    if range == '[0,1]':
        if mode == 'sub':
            return (x - x_min) / (x_max - x_min)
        elif mode == 'add':
            return x * (x_max - x_min) + x_min
        else:
            raise NotImplementedError

    elif range == '[-1,1]':
        if mode == 'sub':
            return 2.0 * ((x - x_min) / (x_max - x_min)) - 1.0
        elif mode == 'add':
            x = (x + 1.0) / 2.0
            return x * (x_max - x_min) + x_min
        elif mode == 'std':
            return (x - np.mean(x)) / np.std(x)        
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
