#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug spectrum data structure"""

import numpy as np

def parse_spe_file(filepath):
    metadata = {}
    spectrum = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    in_data = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('$DATA:'):
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
    
    return np.array(spectrum, dtype=np.int64)

# Parse
co60 = parse_spe_file('Co60.Spe')
eu152 = parse_spe_file('Eu152.Spe')

print(f"Co60 spectrum shape: {co60.shape}")
print(f"Eu152 spectrum shape: {eu152.shape}")

# Show statistics
print("\n=== Co60 Statistics ===")
print(f"Max: {np.max(co60):,}")
print(f"Min: {np.min(co60):,}")
print(f"Mean: {np.mean(co60):,.0f}")
print(f"Std: {np.std(co60):,.0f}")

# Show first values
print("\nFirst 50 Co60 values:")
for i in range(0, min(50, len(co60)), 5):
    print(f"  [{i}:{min(i+5, len(co60))}]: {co60[i:min(i+5, len(co60))]}")

# Find where significant values start
significant_threshold = np.max(co60) * 0.01
idx_significant = np.where(co60 > significant_threshold)[0]
if len(idx_significant) > 0:
    print(f"\nSignificant values (>{significant_threshold:,.0f}) start at channel: {idx_significant[0]}")
    print(f"End at channel: {idx_significant[-1]}")
    print(f"Peak height region: channels {idx_significant[0]} to {idx_significant[-1]}")

# Look for local maxima in the significant region
print("\n=== Finding local maxima ===")
maxima = []
for i in range(10, len(co60) - 10):
    if (co60[i] > co60[i-1] and co60[i] > co60[i+1] and
        co60[i] > co60[i-2] and co60[i] > co60[i+2] and
        co60[i] > co60[i-5] and co60[i] > co60[i+5]):
        maxima.append((i, co60[i]))

print(f"Found {len(maxima)} local maxima")
for ch, val in sorted(maxima, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  Channel {ch}: {val:,}")

# Derivative analysis
print("\n=== Derivative analysis ===")
deriv = np.diff(co60)
print(f"Max derivative: {np.max(deriv)}")
print(f"Min derivative: {np.min(deriv)}")

# Find turning points (peaks)
peaks_deriv = []
for i in range(1, len(deriv)-1):
    if deriv[i-1] > 0 and deriv[i] < 0:  # Local maximum
        peaks_deriv.append(i+1)
        
print(f"Found {len(peaks_deriv)} peaks from derivative analysis")
for ch in sorted(peaks_deriv, key=lambda x: co60[x], reverse=True)[:10]:
    print(f"  Channel {ch}: {co60[ch]:,}")

print(f"\nTotal data points: {len(co60)}")
