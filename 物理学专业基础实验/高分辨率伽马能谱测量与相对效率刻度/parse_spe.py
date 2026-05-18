#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse Canberra .Spe gamma-ray spectrum files and process data
"""

import re
import numpy as np
import json

def parse_spe_file(filepath):
    """Parse a Canberra .Spe file and return metadata and spectrum data"""
    metadata = {}
    spectrum = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse header metadata
    i = 0
    in_data_section = False
    data_range = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('$SPEC_ID:'):
            metadata['spec_id'] = lines[i+1].strip() if i+1 < len(lines) else ''
            i += 1
        elif line.startswith('$SPEC_REM:'):
            metadata['remarks'] = lines[i+1].strip() if i+1 < len(lines) else ''
            i += 1
        elif line.startswith('$DETDESC#'):
            metadata['detector'] = line.split('DETDESC#')[1].strip()
        elif line.startswith('$DATE_MEA:'):
            metadata['date_measured'] = lines[i+1].strip() if i+1 < len(lines) else ''
            i += 1
        elif line.startswith('$MEAS_TIM:'):
            times = lines[i+1].strip().split()
            if len(times) >= 2:
                metadata['live_time'] = float(times[0])
                metadata['real_time'] = float(times[1])
            i += 1
        elif line.startswith('$DATA:'):
            data_line = lines[i+1].strip().split()
            if len(data_line) >= 2:
                data_range = (int(data_line[0]), int(data_line[1]))
                metadata['data_range'] = data_range
            in_data_section = True
            i += 1
        elif in_data_section and line and not line.startswith('$'):
            try:
                spectrum.append(int(line))
            except ValueError:
                pass
        elif line.startswith('$'):
            in_data_section = False
        
        i += 1
    
    spectrum = np.array(spectrum)
    
    # Create channel-to-energy mapping (placeholder - will be calibrated)
    channels = np.arange(len(spectrum))
    
    return metadata, spectrum, channels

def find_peaks(spectrum, height_threshold=None, distance=5):
    """Find peaks in the spectrum using simple local maxima detection"""
    # Use a dynamic threshold if not specified
    if height_threshold is None:
        height_threshold = np.mean(spectrum) + 2 * np.std(spectrum)
    
    # Find local maxima
    peaks = []
    for i in range(distance, len(spectrum) - distance):
        if spectrum[i] > height_threshold:
            is_local_max = True
            for j in range(1, distance + 1):
                if spectrum[i-j] >= spectrum[i] or spectrum[i+j] >= spectrum[i]:
                    is_local_max = False
                    break
            if is_local_max:
                peaks.append(i)
    
    peaks = np.array(peaks)
    properties = {'prominences': spectrum[peaks]} if len(peaks) > 0 else {'prominences': []}
    
    return peaks, properties

def calibrate_energy_from_co60(spectrum_co60, co60_energies=(1173, 1332)):
    """
    Calibrate energy using Co60 peaks
    Returns: channel-to-energy mapping parameters (a, b) where Energy = a * channel + b
    """
    # Find the two strongest peaks in Co60 spectrum
    peaks, props = find_peaks(spectrum_co60, height=np.max(spectrum_co60)*0.1)
    
    if len(peaks) >= 2:
        # Get the two strongest peaks
        peak_indices = peaks[np.argsort(spectrum_co60[peaks])[-2:]]
        peak_channels = peak_indices[np.argsort(peak_indices)]  # Sort by position
        
        # Fit linear energy calibration
        # Assuming the lower channel is 1173 keV and higher is 1332 keV
        x = peak_channels
        y = np.array(co60_energies)
        
        # Linear fit: E = a*channel + b
        coeffs = np.polyfit(x, y, 1)
        
        return coeffs
    
    return None

def background_subtraction(spectrum, peak_channels, peak_width=5):
    """
    Subtract background from a peak
    peak_channels: array or list of channel indices for the peak
    """
    min_ch = min(peak_channels)
    max_ch = max(peak_channels)
    
    # Define background regions (left and right of peak)
    bg_left_start = max(0, min_ch - peak_width * 3)
    bg_left_end = max(0, min_ch - peak_width)
    bg_right_start = max_ch + peak_width
    bg_right_end = min(len(spectrum), max_ch + peak_width * 3)
    
    # Calculate background levels
    if bg_left_end > bg_left_start:
        bg_left = np.mean(spectrum[bg_left_start:bg_left_end])
    else:
        bg_left = spectrum[min_ch]
    
    if bg_right_end > bg_right_start:
        bg_right = np.mean(spectrum[bg_right_start:bg_right_end])
    else:
        bg_right = spectrum[max_ch]
    
    # Linear interpolation of background
    n_channels = max_ch - min_ch + 1
    bg_channels = np.linspace(bg_left, bg_right, n_channels)
    
    # Subtract background
    peak_spectrum = spectrum[min_ch:max_ch+1].copy().astype(float)
    peak_spectrum -= bg_channels
    peak_spectrum = np.maximum(peak_spectrum, 0)  # Ensure non-negative
    
    return peak_spectrum, min_ch, max_ch

def calculate_fwhm(spectrum, peak_channel):
    """Calculate Full Width at Half Maximum (FWHM) of a peak"""
    peak_height = spectrum[peak_channel]
    half_height = peak_height / 2
    
    # Find left half-maximum
    left_idx = peak_channel
    while left_idx > 0 and spectrum[left_idx] > half_height:
        left_idx -= 1
    
    # Find right half-maximum
    right_idx = peak_channel
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_height:
        right_idx += 1
    
    fwhm_channels = right_idx - left_idx
    
    return fwhm_channels

def peak_to_compton_ratio(spectrum, peak_channels, compton_range):
    """
    Calculate peak-to-compton ratio
    peak_channels: channels in the peak
    compton_range: (start_channel, end_channel) for compton plateau
    """
    peak_area = np.sum(spectrum[min(peak_channels):max(peak_channels)+1])
    
    compton_start, compton_end = compton_range
    compton_avg = np.mean(spectrum[compton_start:compton_end+1])
    compton_area = compton_avg * (compton_end - compton_start + 1)
    
    if compton_area > 0:
        ratio = peak_area / compton_area
    else:
        ratio = 0
    
    return ratio

def efficiency_function(E, c1, c2, c3):
    """Relative efficiency function: ln(p) = c1 + c2*ln(E) + c3*(ln(E))^2"""
    ln_E = np.log(E)
    ln_p = c1 + c2 * ln_E + c3 * ln_E**2
    return np.exp(ln_p)

def fit_efficiency(energies, areas, branch_ratios):
    """
    Fit relative efficiency using formula: ln(p) = c1 + c2*ln(E) + c3*(ln(E))^2
    
    Args:
        energies: array of peak energies in MeV
        areas: array of peak areas (counts)
        branch_ratios: array of branching ratios (%)
    
    Returns:
        Fitted parameters [c1, c2, c3] and efficiency values
    """
    # Normalize by branching ratio to get relative intensity
    normalized_areas = areas / (branch_ratios / 100.0)
    
    # Use the lowest energy as reference (efficiency = 1)
    min_energy_idx = np.argmin(energies)
    reference_area = normalized_areas[min_energy_idx]
    relative_efficiency = normalized_areas / reference_area
    
    # Fit the logarithmic function using polynomial fit on log-transformed data
    ln_E = np.log(energies)
    ln_efficiency = np.log(relative_efficiency)
    
    # Create design matrix for polynomial fit: ln(p) = c1 + c2*ln(E) + c3*(ln(E))^2
    A = np.column_stack([np.ones_like(ln_E), ln_E, ln_E**2])
    
    try:
        # Least squares fit
        popt, _, _, _ = np.linalg.lstsq(A, ln_efficiency, rcond=None)
        fitted_efficiency = efficiency_function(energies, *popt)
        
        return popt, relative_efficiency, fitted_efficiency
    except Exception as e:
        print(f"Error in fitting: {e}")
        return None, None, None

# Main processing
if __name__ == '__main__':
    # Parse Co60 data
    print("Parsing Co60 spectrum...")
    meta_co60, spec_co60, ch_co60 = parse_spe_file('Co60.Spe')
    print(f"Co60 metadata: {meta_co60}")
    print(f"Co60 spectrum length: {len(spec_co60)}")
    
    # Parse Eu152 data
    print("\nParsing Eu152 spectrum...")
    meta_eu152, spec_eu152, ch_eu152 = parse_spe_file('Eu152.Spe')
    print(f"Eu152 metadata: {meta_eu152}")
    print(f"Eu152 spectrum length: {len(spec_eu152)}")
    
    # Find peaks in both spectra
    print("\nFinding peaks in Co60...")
    peaks_co60, props_co60 = find_peaks(spec_co60, height_threshold=np.max(spec_co60)*0.05)
    print(f"Found {len(peaks_co60)} peaks in Co60")
    print(f"Top 5 peak channels: {peaks_co60[np.argsort(spec_co60[peaks_co60])[-5:]]}")
    
    print("\nFinding peaks in Eu152...")
    peaks_eu152, props_eu152 = find_peaks(spec_eu152, height_threshold=np.max(spec_eu152)*0.05)
    print(f"Found {len(peaks_eu152)} peaks in Eu152")
    print(f"Top 10 peak channels: {peaks_eu152[np.argsort(spec_eu152[peaks_eu152])[-10:]]}")
    
    # Estimate energy calibration from Co60 1173 and 1332 keV peaks
    print("\nEstimating energy calibration from Co60...")
    # These will be the two strongest peaks
    strongest_peaks = peaks_co60[np.argsort(spec_co60[peaks_co60])[-2:]]
    print(f"Strongest peaks at channels: {sorted(strongest_peaks)}")
    print(f"Peak heights: {spec_co60[sorted(strongest_peaks)]}")
    
    print("\nData processing complete!")
    print(f"Co60 real time: {meta_co60.get('real_time', 'N/A')} s")
    print(f"Eu152 real time: {meta_eu152.get('real_time', 'N/A')} s")
