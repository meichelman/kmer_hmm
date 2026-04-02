import numpy as np
import os
from collections import defaultdict

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions for handling observertions/bed files
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_obs_mut(obs_file, mutrates_file, window_size):
    # Build contig -> global obs index offset map AND per-contig window count
    contig_offsets = {}
    contig_window_counts = defaultdict(int)
    offset = 0

    with open(obs_file) as infile:
        for line in infile:
            contig = line.split('\t')[0]
            if contig not in contig_offsets:
                contig_offsets[contig] = offset
            contig_window_counts[contig] += 1
            offset += 1

    num_windows = offset
    obs_arr = np.zeros(num_windows, dtype=np.int16)

    with open(obs_file) as infile:
        for idx, line in enumerate(infile):
            count = line.split('\t')[3]
            obs_arr[idx] = int(count)

    mutrates_arr = np.zeros(num_windows, dtype=float)

    with open(mutrates_file) as infile:
        for line in infile:
            if line.startswith('contig'):
                continue
            contig, start, end, mutrate = line.strip().split('\t')
            start, end, mutrate = int(start), int(end), float(mutrate)

            if contig not in contig_offsets:
                continue

            contig_offset = contig_offsets[contig]
            contig_max_idx = contig_offset + contig_window_counts[contig]  # true end for this contig

            obs_idx_start = contig_offset + start // window_size
            obs_idx_end   = contig_offset + end // window_size
            obs_idx_end   = min(obs_idx_end, contig_max_idx)  # cap at contig boundary, not global

            mutrates_arr[obs_idx_start:obs_idx_end] = mutrate

    for idx, (obs, m) in enumerate(zip(obs_arr, mutrates_arr)):
        if obs > 0 and m == 0:
            # find which contig this index belongs to
            for contig, c_offset in contig_offsets.items():
                if c_offset <= idx < c_offset + contig_window_counts[contig]:
                    local_idx = idx - c_offset
                    print(f"Warning: observation={obs} but mutrate=0 at index={idx} "
                        f"| contig={contig} local_idx={local_idx} "
                        f"| contig_windows={contig_window_counts[contig]} "
                        f"| approx_pos={local_idx * window_size}")
                    break

    return obs_arr, mutrates_arr

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
