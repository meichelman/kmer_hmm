import argparse
from hmm_functions import read_HMM_parameters_from_file, emission_probabilities, Calculate_Posterior_probabillities, PMAP_path, Viterbi_path, Write_posterior_probs
from helper_functions import load_obs_and_obs_rates
import numpy as np



def decode(obs_file, obs_rates_file, param_file, out_file):
    
    print('Loading data...')
    hmm_parameters = read_HMM_parameters_from_file(param_file)
    obs, obs_rates, contig_offsets, window_size = load_obs_and_obs_rates(obs_file, obs_rates_file)

    print('-' * 40)
    print(f'> Number of windows: {len(obs)}')
    print(f'> Number of k-mers: {obs.astype(np.int64).sum()}')
    print('-' * 40)

    print('Calculating posterior probabilities...')
    emissions_probs = emission_probabilities(obs, obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)
    posterior_probs = Calculate_Posterior_probabillities(emissions_probs, hmm_parameters)
    print('Determining most likely path using posterior probabilities...')
    pmap_path = PMAP_path(posterior_probs)
    print('Determining most likely path using Viterbi algorithm...')
    viterbi_path = Viterbi_path(emissions_probs, hmm_parameters)
    print('Writing output...')
    Write_posterior_probs(obs, obs_rates, posterior_probs, pmap_path, viterbi_path, hmm_parameters, out_file, window_size)
    print('Done')
    
    return


def print_script_usage():
    toprint = f'''
    Hidden Markov Model for archaic human introgression inference using the ARCkmerFinder output.

    Usage:
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -param [hmm_parameters_file]
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -param [hmm_parameters_file] -out [output_file]
        
    > HMM decoding                
        -obs                Input file with observation data (required)
        -obs_rates          Input file with observation rates estimates (required)
        -param              HMM parameters file (required)
        -out                Output file with decoded paths (default: 'paths.txt')
    '''

    return toprint
    

def main():
    parser = argparse.ArgumentParser(description=print_script_usage(), formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-obs",help="Input file with observation data (required)", type=str)
    parser.add_argument("-obs_rates", metavar='',help="Input file with observation rates estimates (required)", type=str)
    parser.add_argument("-param", metavar='',help="HMM parameters file (required)", type=str)
    parser.add_argument("-out", metavar='',help="Output file with decoded paths (default: 'paths.txt')", default = 'paths.txt')
    
    args = parser.parse_args()
    
    if args.obs is None or args.obs_rates is None or args.param is None:
        print(print_script_usage())
        return
    
    print('-' * 40)
    print(f'> Observations file: {args.obs}')
    print(f'> Observation rates file: {args.obs_rates}')
    print(f'> HMM parameters file: {args.param}')
    print(f'> Output file with decoded paths: {args.out}')
    print('-' * 40)

    decode(args.obs, args.obs_rates, args.param, args.out)
    
    return
        

if __name__ == "__main__":
    main()
