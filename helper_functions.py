import numpy as np
import os
from collections import defaultdict
import math



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions for handling observations
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_obs_and_obs_rates(obs_file, obs_rates_file):
    
    # Build contig -> global obs index offset map AND per-contig window count
    contig_offsets = {}
    contig_window_counts = defaultdict(int)
    offset = 0
    window_size = None
    with open(obs_file) as infile:
        for line in infile:
            fields = line.split('\t')
            contig = fields[0]
            if window_size is None:
                window_size = int(fields[2]) - int(fields[1])
            if contig not in contig_offsets:
                contig_offsets[contig] = offset
            contig_window_counts[contig] += 1
            offset += 1

    num_windows = offset
    obs_arr = np.zeros(num_windows, dtype=np.int16)
    obs_rates_arr = np.zeros(num_windows, dtype=float)

    # Load obs counts into array, using contig_offsets to determine global index
    with open(obs_file) as infile:
        for idx, line in enumerate(infile):
            obs_arr[idx] = int(line.split('\t')[3])

    # Load obs_rates, mapping each rate to its windows by genomic position
    with open(obs_rates_file) as infile:
        for line in infile:
            if line.startswith('contig'):
                continue
            fields = line.strip().split('\t')
            contig, start, end, obs_rate = fields[0], int(fields[1]), int(fields[2]), float(fields[3])

            contig_offset = contig_offsets[contig]
            contig_end = contig_offset + contig_window_counts[contig]

            global_idx_start = contig_offset + start // window_size
            global_idx_end   = min(contig_offset + math.ceil(end / window_size), contig_end)

            obs_rates_arr[global_idx_start:global_idx_end] = obs_rate

    # Catch errors involving the observation and observation rate not aligning
    for idx, (obs, obs_rate) in enumerate(zip(obs_arr, obs_rates_arr)):
        if obs > 0 and obs_rate == 0:
            for contig, c_offset in contig_offsets.items():
                if c_offset <= idx < c_offset + contig_window_counts[contig]:
                    local_idx = idx - c_offset
                    print(f"Warning: observation={obs} but obs_rate=0 at index={idx} "
                        f"| contig={contig} local_idx={local_idx} "
                        f"| contig_windows={contig_window_counts[contig]} "
                        f"| approx_pos={local_idx * window_size}")
                    break

    return obs_arr, obs_rates_arr, contig_offsets, window_size


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Various helper functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def Make_folder_if_not_exists(path):
    '''
    Check if path exists - otherwise make it;
    '''
    path = os.path.dirname(path)
    if path != '':
        if not os.path.exists(path):
            os.makedirs(path)
