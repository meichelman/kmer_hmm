import argparse
from hmm_functions import read_HMM_parameters_from_file, emission_probabilities, calculate_posterior_probabilities, pmap_path, viterbi_path, write_posterior_probs, write_tracts
from helper_functions import load_obs_and_obs_rates
import numpy as np



def decode(obs_file, obs_rates_file, param_file, decode_method, out_path_file, out_tracts_file):
    
    print('Loading data...')
    hmm_parameters = read_HMM_parameters_from_file(param_file)
    obs, obs_rates, contig_offsets, window_size = load_obs_and_obs_rates(obs_file, obs_rates_file)

    print('-' * 40)
    print(f'> Number of windows: {len(obs)}')
    print(f'> Number of k-mers: {obs.astype(np.int64).sum()}')
    print('-' * 40)
    
    # Build contig slices from offsets
    contig_list = list(contig_offsets.keys())
    contig_slices = {}
    for i, contig in enumerate(contig_list):
        start = contig_offsets[contig]
        end = contig_offsets[contig_list[i + 1]] if i + 1 < len(contig_list) else len(obs)
        contig_slices[contig] = slice(start, end)
        
    all_obs = []
    all_obs_rates = []
    all_posterior_probs = []
    all_paths = []
    
    print('Calculating posterior probabilities...')
    for contig, sl in contig_slices.items():
        contig_obs = obs[sl]
        contig_obs_rates = obs_rates[sl]

        emissions_probs = emission_probabilities(contig_obs, contig_obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)
        posterior_probs = calculate_posterior_probabilities(emissions_probs, hmm_parameters)

        if decode_method == 'Viterbi':
            path = viterbi_path(emissions_probs, hmm_parameters)
        else:
            path = pmap_path(posterior_probs)

        all_obs.append(contig_obs)
        all_obs_rates.append(contig_obs_rates)
        all_posterior_probs.append(posterior_probs)
        all_paths.append(path)

    print('Writing output...')
    write_posterior_probs(
        contig_slices,
        np.concatenate(all_obs),
        np.concatenate(all_obs_rates),
        np.concatenate(all_posterior_probs, axis=-1),
        np.concatenate(all_paths),
        hmm_parameters,
        out_path_file,
        window_size
    )
    write_tracts(
        contig_slices,
        np.concatenate(all_paths),
        hmm_parameters,
        out_tracts_file,
        window_size
    )

    print('Done')
    
    return


def print_script_usage():
    toprint = f'''
    Hidden Markov Model for archaic human introgression inference using the ARCkmerFinder output.

    Usage:
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -param [hmm_parameters_file]
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -param [hmm_parameters_file] -viterbi -out_path [out_path_file] -out_tracts [out_tracts_file] 
        
    > HMM decoding                
        -obs                Input file with observation data (required)
        -obs_rates          Input file with observation rates estimates (required)
        -param              HMM parameters file (required)
        -out_path           Output file with decoded path (default: 'path.txt')
        -out_tracts         Output file with decoded tracts (default: 'tracts.txt')
    '''

    return toprint
    

def main():
    parser = argparse.ArgumentParser(description=print_script_usage(), formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-obs",help="Input file with observation data (required)", type=str)
    parser.add_argument("-obs_rates", metavar='',help="Input file with observation rates estimates (required)", type=str)
    parser.add_argument("-param", metavar='',help="HMM parameters file (required)", type=str)
    parser.add_argument("-viterbi", action='store_true', help="Decode using Viterbi algorithm (default: False)")
    parser.add_argument("-out_path", metavar='',help="Output file with decoded path (default: 'path.txt')", default = 'path.txt')
    parser.add_argument("-out_tracts", metavar='',help="Output file with decoded tracts (default: 'tracts.txt')", default = 'tracts.txt')

    args = parser.parse_args()
    
    if args.obs is None or args.obs_rates is None or args.param is None:
        print(print_script_usage())
        return
    
    print('-' * 40)
    print(f'> Observations file: {args.obs}')
    print(f'> Observation rates file: {args.obs_rates}')
    print(f'> HMM parameters file: {args.param}')
    if args.viterbi:
        decode_method = 'Viterbi'
        print('> Decoding method: Viterbi algorithm')
    else:
        decode_method = 'PMAP'
        print('> Decoding method: PMAP')
    print(f'> Output file with decoded paths: {args.out}')
    print('-' * 40)

    decode(args.obs, args.obs_rates, args.param, decode_method, args.out_path, args.out_tracts)
    
    return
        

if __name__ == "__main__":
    main()
