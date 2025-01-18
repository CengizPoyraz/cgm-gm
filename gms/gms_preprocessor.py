import numpy as np
import pandas as pd
import torch
import torchaudio
from typing import Tuple, Dict, Optional
import logging
class WaveformPreprocessor:
    """Preprocessing class for earthquake waveform data."""
    
    def __init__(self, 
                 window_length: int = 160,
                 hop_length: int = 46,
                 n_fft: int = 160,
                 power: int = 1,
                 min_freq: float = 2.0,
                 max_freq: float = 15.0,
                 min_snr: float = 3.0,
                 device: str = 'cuda'):
        """
        Initialize preprocessor with parameters.
        
        Args:
            window_length: STFT window length
            hop_length: STFT hop length
            n_fft: Number of FFT points
            power: Spectrogram power
            min_freq: Minimum frequency for filtering (Hz)
            max_freq: Maximum frequency for filtering (Hz) 
            min_snr: Minimum signal-to-noise ratio
            device: Computing device ('cuda' or 'cpu')
        """
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.power = power
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_snr = min_snr
        self.device = device
        
        # Initialize STFT converter
        self.time_spec_converter = TimeSpecConverter(
            n_fft=n_fft,
            window_length=window_length,
            hop_length=hop_length,
            power=power,
            device=device
        )

    def compute_snr(self, waveform: np.ndarray, fs: float) -> float:
        """
        Compute signal-to-noise ratio in frequency domain.
        
        Args:
            waveform: Input waveform [samples]
            fs: Sampling frequency
            
        Returns:
            SNR value
        """
        # Split into noise and signal portions
        noise = waveform[:1000]  # First 10s at 100Hz
        signal = waveform[1000:]  # Remaining signal
        
        # Compute FFT
        noise_fft = np.abs(np.fft.rfft(noise))
        signal_fft = np.abs(np.fft.rfft(signal))
        
        # Get frequency bins
        freqs = np.fft.rfftfreq(len(noise), d=1/fs)
        
        # Find indices within frequency range
        freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        # Compute mean SNR in frequency range
        snr = np.mean(signal_fft[freq_mask] / noise_fft[freq_mask])
        
        return snr

    def normalize_waveform(self, waveform: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize waveform to [0,1] range.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Normalized waveform and normalization parameters
        """
        wf_min = waveform.min()
        wf_max = waveform.max()
        
        normalized = (waveform - wf_min) / (wf_max - wf_min)
        
        norm_dict = {
            'wf_min': wf_min,
            'wf_max': wf_max
        }
        
        return normalized, norm_dict

    def process_waveform(self, 
                        waveform: np.ndarray,
                        sampling_freq: float,
                        validate: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full preprocessing pipeline for a single waveform.
        
        Args:
            waveform: Raw waveform data
            sampling_freq: Sampling frequency
            validate: Whether to validate SNR
            
        Returns:
            Processed amplitude, phase and normalization parameters
        """
        # Validate SNR if requested
        if validate:
            snr = self.compute_snr(waveform, sampling_freq)
            if snr < self.min_snr:
                raise ValueError(f"SNR {snr:.2f} below minimum {self.min_snr}")
        
        # Convert to tensor
        wf_tensor = torch.from_numpy(waveform).float().to(self.device)
        if len(wf_tensor.shape) == 1:
            wf_tensor = wf_tensor.unsqueeze(0)
        
        # Apply STFT
        spectrogram = self.time_spec_converter.time_to_spec(wf_tensor)
        
        # Separate magnitude and phase
        phase = torch.angle(spectrogram)
        magnitude = torch.abs(spectrogram)
        
        # Add small constant before log transform
        eps = 1e-10
        magnitude = magnitude + eps
        magnitude = torch.log10(magnitude)
        
        # Normalize magnitude
        magnitude, norm_dict = self.normalize_waveform(magnitude.cpu().numpy())
        magnitude = torch.from_numpy(magnitude).to(self.device)
        
        # Permute dimensions for model input
        magnitude = magnitude.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)
        
        return magnitude, phase, norm_dict

    def batch_process(self, 
                     waveforms: np.ndarray,
                     sampling_freq: float,
                     validate: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process a batch of waveforms.
        
        Args:
            waveforms: Batch of raw waveforms [batch, samples]
            sampling_freq: Sampling frequency
            validate: Whether to validate SNR
            
        Returns:
            Processed amplitudes, phases and normalization parameters
        """
        magnitudes = []
        phases = []
        norm_dicts = []
        
        for i in range(len(waveforms)):
            try:
                mag, phase, norm_dict = self.process_waveform(
                    waveforms[i],
                    sampling_freq,
                    validate
                )
                magnitudes.append(mag)
                phases.append(phase)
                norm_dicts.append(norm_dict)
            except Exception as e:
                logging.warning(f"Failed to process waveform {i}: {str(e)}")
                continue
                
        # Stack tensors
        magnitudes = torch.cat(magnitudes, dim=0)
        phases = torch.cat(phases, dim=0)
        
        # Combine norm dicts
        combined_dict = {
            'wf_min': min(d['wf_min'] for d in norm_dicts),
            'wf_max': max(d['wf_max'] for d in norm_dicts)
        }
        
        return magnitudes, phases, combined_dict


class TimeSpecConverter:
    """Handles conversions between time and frequency domains."""
    
    def __init__(self,
                 n_fft: int,
                 window_length: int,
                 hop_length: int, 
                 power: int,
                 device: str,
                 n_iter: int = 50):
        """
        Initialize converter.
        
        Args:
            n_fft: Number of FFT points
            window_length: Analysis window length
            hop_length: Number of samples between windows
            power: Spectrogram power
            device: Computing device
            n_iter: Griffin-Lim iterations
        """
        self.n_fft = n_fft
        self.window_length = window_length 
        self.hop_length = hop_length
        self.power = power
        self.device = device
        self.n_iter = n_iter
        
        self.griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=window_length,
            hop_length=hop_length,
            power=power,
            n_iter=n_iter
        ).to(device)
        
    def time_to_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert time domain signal to spectrogram."""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            return_complex=True
        )
    
    def spec_to_time(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to time domain signal."""
        return torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length
        )

def load_and_preprocess_data(data_path, preprocessor):
    """
    Load and preprocess waveform data
    
    Args:
        data_path: Path to CSV file
        preprocessor: WaveformPreprocessor instance
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Extract sampling frequency from data (column 7)
    sampling_freq = df.iloc[0, 7]  # Get sampling frequency from first row
    print(f"Sampling frequency: {sampling_freq} Hz")
    
    # Get waveform data starting from column 14
    print("Extracting waveforms...")
    waveforms = df.iloc[:, 14:].values
    
    print(f"Waveform data shape: {waveforms.shape}")
    print(f"Waveform sample: \n{waveforms[0, :10]}")  # Print first 10 values of first waveform
    
    # Reshape if needed - WaveformPreprocessor expects [samples] or [batch, samples]
    if len(waveforms.shape) == 2:
        print("Reshaping waveforms to [batch, samples]")
        waveforms = waveforms.reshape(waveforms.shape[0], -1)
        
    print(f"Waveform data range: [{waveforms.min()}, {waveforms.max()}]")
    print(f"Any NaN values: {np.isnan(waveforms).any()}")
    
    # Process waveforms in smaller batches to avoid memory issues
    batch_size = 100
    total_samples = len(waveforms)
    all_magnitudes = []
    all_phases = []
    all_norm_dicts = []
    
    print(f"\nProcessing {total_samples} waveforms in batches of {batch_size}")
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch = waveforms[i:end_idx]
        
        try:
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            print(f"Batch shape: {batch.shape}")
            
            magnitudes, phases, norm_dict = preprocessor.batch_process(
                batch,
                sampling_freq=sampling_freq,
                validate=False  # Set to True if you want SNR validation
            )
            
            if magnitudes is not None and len(magnitudes) > 0:
                all_magnitudes.append(magnitudes)
                all_phases.append(phases)
                all_norm_dicts.append(norm_dict)
                print(f"Successfully processed batch. Magnitudes shape: {magnitudes.shape}")
            else:
                print("Warning: Empty or None output from batch_process")
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue
    
    if not all_magnitudes:
        raise RuntimeError("No waveforms were successfully processed")
    
    # Combine results
    try:
        combined_magnitudes = torch.cat(all_magnitudes, dim=0)
        combined_phases = torch.cat(all_phases, dim=0)
        
        # Combine normalization dictionaries
        combined_norm_dict = {
            'wf_min': min(d['wf_min'] for d in all_norm_dicts),
            'wf_max': max(d['wf_max'] for d in all_norm_dicts)
        }
        
        print(f"\nFinal processed data:")
        print(f"Magnitudes shape: {combined_magnitudes.shape}")
        print(f"Phases shape: {combined_phases.shape}")
        print(f"Normalization range: [{combined_norm_dict['wf_min']}, {combined_norm_dict['wf_max']}]")
        
        return combined_magnitudes, combined_phases, combined_norm_dict
        
    except Exception as e:
        print("Error combining results:")
        print(f"Number of magnitude tensors: {len(all_magnitudes)}")
        print(f"Number of phase tensors: {len(all_phases)}")
        print(f"Error message: {str(e)}")
        raise
   