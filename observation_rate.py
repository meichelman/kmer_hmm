import numpy as np
import argparse
from collections import defaultdict
from helper_functions import Make_folder_if_not_exists

        
    
def make_obs_rate(obs_file, out_file, bin_size):
    
    print('Estimating observation rates...')
    
    # Determine contig offsets and window counts
    print('Calculating contig offsets and window counts...')
    kmer_dict = defaultdict(lambda: defaultdict(int))
    contig_lengths = defaultdict(int)
    with open(obs_file) as infile:
        for line in infile:
            contig, start, end, count = line.split('\t')
            start, end, count = int(start), int(end), int(count)
            
            if count == 0:
                # Span covers multiple zero windows
                for window in range(start - start % bin_size, end, bin_size):
                    kmer_dict[contig][window] += 0
            else:
                window = start - start % bin_size
                kmer_dict[contig][window] += count

            contig_lengths[contig] = max(contig_lengths[contig], end)

    # Calculate observation rates
    print('Calculating observation rates...')
    kmer_arr = []
    assembly_positions = []
    for contig in kmer_dict:
        last_window = max(kmer_dict[contig]) + bin_size
        last_window = int(last_window)

        for window in range(0, last_window, bin_size):
            kmer_arr.append(kmer_dict[contig][window])
            # Store actual window end, capped at contig length
            actual_end = min(window + bin_size, contig_lengths[contig])
            assembly_positions.append([contig, window, actual_end])

    # Calculate average k-mer count per base across the assembly
    kmer_arr = np.array(kmer_arr)
    assembly_length = sum(contig_lengths.values())
    assembly_avg = np.sum(kmer_arr) / assembly_length

    # Write output
    print('Writing output...')
    with open(out_file, 'w') as outfile:
        outfile.write('contig\tstart\tend\tobs_rate\n')
        for position, kmer_count in zip(assembly_positions, kmer_arr):
            contig, start, end = position
            actual_window_size = end - start  # use real span, not nominal window_size
            obs_rate = round(kmer_count / actual_window_size / assembly_avg, 5)
            outfile.write(f'{contig}\t{start}\t{end}\t{obs_rate}\n')
            
    print('Done')
            
    return


def print_script_usage():
    toprint = f'''
    Hidden Markov Model for archaic human introgression inference using the ARCkmerFinder output.

    Usage:
    python observation_rate.py -obs [obs_file]
    python observation_rate.py -obs [obs_file] -out [output_file] -bin_size [bin_size]
    
    > Estimate observation rate
        -obs            Input file with observation data (required)
        -out            Output file with observation rate estimates (default: 'obs_rate.bed')
        -bin_size       Parameter defining size of bins (default: 1 Mb)
        
    '''

    return toprint


def main():
    parser = argparse.ArgumentParser(description=print_script_usage(), formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-obs",help="Input file with observation data (required)", type=str)
    parser.add_argument("-out", metavar='',help="Output file with observation rates estimates (default: 'obs_rate.bed')", default = 'obs_rate.bed')
    parser.add_argument("-bin_size", metavar='',help="Parameter defining size of bins (default: 1 Mb)", type=int, default = 1_000_000)
    
    args = parser.parse_args()
    
    if args.obs is None:
        print(print_script_usage())
        return
    
    print('-' * 40)
    print(f'> Observations file: {args.obs}')
    print(f'> Output file for estimated observation rates: {args.out}')
    print(f'> Bin size: {args.bin_size}')
    print('-' * 40)

    make_obs_rate(args.obs, args.out, args.bin_size)
        
    return


if __name__ == "__main__":
    main()

