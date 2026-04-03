import numpy as np
from collections import defaultdict

from helper_functions import Make_folder_if_not_exists

def make_mutation_rate(obs_file, out_file, window_size):
    kmer_dict = defaultdict(lambda: defaultdict(int))
    contig_lengths = defaultdict(int)
    with open(obs_file) as data:
        for line in data:
            contig, start, end, count = line.split('\t')
            window = int(start) - int(start) % window_size
            kmer_dict[contig][window] += int(count)
            contig_lengths[contig] = max(contig_lengths[contig], int(end))

    kmer_arr = []
    assembly_positions = []
    for contig in kmer_dict:
        lastwindow = max(kmer_dict[contig]) + window_size
        lastwindow = int(lastwindow)

        for window in range(0, lastwindow, window_size):
            kmer_arr.append(kmer_dict[contig][window])
            # store actual window end, capped at contig length
            actual_end = min(window + window_size, contig_lengths[contig])
            assembly_positions.append([contig, window, actual_end])

    kmer_arr = np.array(kmer_arr)
    assembly_length = sum(contig_lengths.values())
    assembly_avg = np.sum(kmer_arr) / assembly_length
    # print(f'> assembly average = {assembly_avg}')

    Make_folder_if_not_exists(out_file)
    with open(out_file, 'w') as out:
        out.write('contig\tstart\tend\tmutationrate\n')
        for pos, kmer_count in zip(assembly_positions, kmer_arr):
            contig, start, end = pos
            actual_window_size = end - start  # use real span, not nominal window_size
            # mutrate = round(kmer_count / actual_window_size / assembly_avg, 5)
            mutrate = round(assembly_avg, 5)
            out.write(f'{contig}\t{start}\t{end}\t{mutrate}\n')