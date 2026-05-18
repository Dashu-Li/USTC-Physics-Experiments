#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive gamma-ray spectrum analysis for Co60 and Eu152
"""

import numpy as np
import json

def parse_spe_file_detailed(filepath):
    """Parse Canberra .Spe file with detailed analysis"""
    metadata = {}
    spectrum = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse header
    i = 0
    in_data = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('$SPEC_ID:'):
            metadata['spec_id'] = lines[i+1].strip() if i+1 < len(lines) else ''
            i += 1
        elif line.startswith('$SPEC_REM:'):
            metadata['remarks'] = lines[i+1].strip() if i+1 < len(lines) else ''
            i += 1
        elif 'DETDESC' in line:
            metadata['detector'] = line.split('DETDESC#')[1].strip() if 'DETDESC#' in line else ''
        elif 'AP#' in line:
            metadata['software'] = line.split('AP#')[1].strip() if 'AP#' in line else ''
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
            data_range = lines[i+1].strip().split()
            if len(data_range) >= 2:
                metadata['data_start'] = int(data_range[0])
                metadata['data_end'] = int(data_range[1])
            in_data = True
            i += 1
        elif in_data and line and not line.startswith('$'):
            try:
                spectrum.append(int(line))
            except ValueError:
                pass
        elif line.startswith('$'):
            in_data = False
        
        i += 1
    
    spectrum = np.array(spectrum, dtype=np.int64)
    return metadata, spectrum

def find_significant_peaks(spectrum, num_peaks=10, smoothing_window=5):
    """Find the N most significant peaks"""
    # Smooth the spectrum slightly to reduce noise
    smoothed = spectrum.copy().astype(float)
    for w in range(smoothing_window):
        smoothed = np.convolve(smoothed, np.ones(3)/3, mode='same')
    
    # Find local maxima
    peaks = []
    for i in range(5, len(smoothed) - 5):
        is_max = True
        for j in range(1, 6):
            if smoothed[i-j] > smoothed[i] or smoothed[i+j] > smoothed[i]:
                is_max = False
                break
        if is_max and smoothed[i] > np.mean(smoothed) * 0.5:
            peaks.append((i, smoothed[i]))
    
    # Sort by height and return top N
    peaks.sort(key=lambda x: x[1], reverse=True)
    top_peaks = [p[0] for p in peaks[:num_peaks]]
    
    return sorted(top_peaks)

def analyze_peak(spectrum, peak_channel, width=20):
    """Analyze a single peak"""
    start = max(0, peak_channel - width)
    end = min(len(spectrum), peak_channel + width + 1)
    
    peak_region = spectrum[start:end]
    peak_height = spectrum[peak_channel]
    
    # Calculate FWHM
    half_height = peak_height / 2
    fwhm_left = peak_channel
    fwhm_right = peak_channel
    
    for i in range(peak_channel, start, -1):
        if spectrum[i] < half_height:
            fwhm_left = i
            break
    
    for i in range(peak_channel, end):
        if spectrum[i] < half_height:
            fwhm_right = i
            break
    
    fwhm_channels = fwhm_right - fwhm_left
    
    # Calculate net area (simple sum for now)
    total_area = np.sum(peak_region)
    
    return {
        'channel': peak_channel,
        'height': peak_height,
        'fwhm_channels': fwhm_channels,
        'area': total_area,
        'left_edge': fwhm_left,
        'right_edge': fwhm_right
    }

def estimate_energy_calibration(co60_peaks_info, known_energies=[1173, 1332]):
    """
    Estimate energy calibration using two Co60 peaks
    Returns: (slope, intercept) for Energy = slope * channel + intercept
    """
    if len(co60_peaks_info) >= 2:
        # Assume the two strongest peaks are at 1173 and 1332 keV
        channels = np.array([co60_peaks_info[0]['channel'], co60_peaks_info[1]['channel']])
        energies = np.array(known_energies, dtype=float)
        
        # Linear calibration
        A = np.column_stack([channels, np.ones_like(channels)])
        coeffs = np.linalg.lstsq(A, energies, rcond=None)[0]
        
        return coeffs[0], coeffs[1]
    
    return None, None

def calculate_compton_ratio(spectrum, peak_channel, peak_width=20, compton_start=None, compton_end=None):
    """Calculate peak-to-Compton ratio"""
    
    # Define peak region
    peak_left = max(0, peak_channel - peak_width)
    peak_right = min(len(spectrum), peak_channel + peak_width + 1)
    peak_area = np.sum(spectrum[peak_left:peak_right])
    
    # Define Compton plateau region (if not specified)
    if compton_start is None:
        # For 1332 keV peak, Compton region is typically around 1040-1090 keV
        # We'll estimate based on approximate channel positions
        compton_start = max(0, peak_channel - 100)
        compton_end = max(0, peak_channel - 50)
    
    if compton_end > compton_start:
        compton_avg = np.mean(spectrum[compton_start:compton_end+1])
        compton_area = compton_avg * (compton_end - compton_start + 1)
        
        if compton_area > 0:
            ratio = peak_area / compton_area
        else:
            ratio = 0
    else:
        ratio = 0
    
    return ratio

# Main analysis
if __name__ == '__main__':
    print("=" * 60)
    print("GAMMA-RAY SPECTRUM ANALYSIS")
    print("=" * 60)
    
    # Parse data
    print("\nParsing Co60 spectrum...")
    meta_co60, spec_co60 = parse_spe_file_detailed('Co60.Spe')
    print(f"  Real time: {meta_co60['real_time']} s")
    print(f"  Spectrum length: {len(spec_co60)} channels")
    print(f"  Max count: {np.max(spec_co60)}")
    
    print("\nParsing Eu152 spectrum...")
    meta_eu152, spec_eu152 = parse_spe_file_detailed('Eu152.Spe')
    print(f"  Real time: {meta_eu152['real_time']} s")
    print(f"  Spectrum length: {len(spec_eu152)} channels")
    print(f"  Max count: {np.max(spec_eu152)}")
    
    # Find peaks
    print("\n" + "=" * 60)
    print("PEAK DETECTION")
    print("=" * 60)
    
    print("\nCo60 peaks:")
    co60_peak_channels = find_significant_peaks(spec_co60, num_peaks=5)
    co60_peaks_info = []
    for ch in co60_peak_channels:
        info = analyze_peak(spec_co60, ch, width=30)
        co60_peaks_info.append(info)
        print(f"  Channel {ch}: Height={info['height']:,}, FWHM={info['fwhm_channels']} channels")
    
    print("\nEu152 peaks (top 8):")
    eu152_peak_channels = find_significant_peaks(spec_eu152, num_peaks=8)
    eu152_peaks_info = []
    for ch in eu152_peak_channels:
        info = analyze_peak(spec_eu152, ch, width=30)
        eu152_peaks_info.append(info)
        print(f"  Channel {ch}: Height={info['height']:,}, Area={info['area']:,}")
    
    # Energy calibration
    print("\n" + "=" * 60)
    print("ENERGY CALIBRATION")
    print("=" * 60)
    
    slope = None
    intercept = None
    
    if len(co60_peaks_info) >= 2:
        slope, intercept = estimate_energy_calibration(co60_peaks_info, [1173, 1332])
        print(f"\nLinear calibration: E [keV] = {slope:.4f} * channel + {intercept:.4f}")
        
        # Convert peak channels to energies
        print("\nCo60 peak energies:")
        for i, info in enumerate(co60_peaks_info[:2]):
            energy = slope * info['channel'] + intercept
            print(f"  Peak {i+1}: Channel {info['channel']} → {energy:.1f} keV")
        
        # Calculate FWHM in keV and energy resolution
        fwhm_1332_channels = co60_peaks_info[1]['fwhm_channels']
        fwhm_1332_keV = fwhm_1332_channels * slope
        print(f"\n1332 keV peak:")
        print(f"  FWHM: {fwhm_1332_channels:.1f} channels = {fwhm_1332_keV:.2f} keV")
        print(f"  Energy resolution: {fwhm_1332_keV / 1332 * 100:.2f}%")
        
        # Calculate peak-to-compton ratio
        ratio = calculate_compton_ratio(spec_co60, co60_peaks_info[1]['channel'], peak_width=20)
        print(f"\n1332 keV peak-to-Compton ratio: {ratio:.2f}")
    else:
        print("\nWarning: Could not calibrate - insufficient peaks found")
    
    # Save results
    results = {
        'metadata': {
            'co60': meta_co60,
            'eu152': meta_eu152
        },
        'co60_peaks': co60_peaks_info,
        'eu152_peaks': eu152_peaks_info,
        'calibration': {
            'slope': float(slope) if slope is not None else None,
            'intercept': float(intercept) if intercept is not None else None
        }
    }
    
    with open('spectrum_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to: spectrum_analysis_results.json")
    print("=" * 60)
