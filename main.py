import sys
from hmm_functions import TrainModel, write_HMM_to_file, read_HMM_parameters_from_file, Calculate_Posterior_probabillities, PMAP_path, Viterbi_path, Write_posterior_probs, Emission_probs
from helper_functions import load_obs_mut
from make_mutationrate import make_mutation_rate
import numpy as np


def train(obs_file, mutrates_file, out_file):
    hmm_parameters = read_HMM_parameters_from_file(None)
    obs, mutrates = load_obs_mut(obs_file, mutrates_file, window_size=2000)

    print('-' * 40)
    print(f'> number of windows: {len(obs)}. Number of kmers = {obs.astype(np.int64).sum()}')
    print('> output is', out_file) 
    print('-' * 40)

    hmm_parameters = TrainModel(obs, mutrates, hmm_parameters)
    write_HMM_to_file(hmm_parameters, out_file)
    

def decode(obs_file, mutrates_file, param_file, out_file):
    hmm_parameters = read_HMM_parameters_from_file(param_file)
    obs, mutrates = load_obs_mut(obs_file, mutrates_file, window_size=2000)

    print('-' * 40)
    print(f'> number of windows: {len(obs)}. Number of kmers = {obs.astype(np.int64).sum()}')
    print('> output is', out_file)
    print('> Decode with posterior decoding')
    print('-' * 40)
    
    emissions = Emission_probs(hmm_parameters.emissions, obs, mutrates)
    posterior_probs = Calculate_Posterior_probabillities(emissions, hmm_parameters)
    pmap_path = PMAP_path(posterior_probs)
    viterbi_path = Viterbi_path(emissions, hmm_parameters)
    
    Write_posterior_probs(obs, mutrates, posterior_probs, pmap_path, viterbi_path, hmm_parameters, out_file, window_size = 2000)


def main():
    args = sys.argv
    if len(args) < 2:
        sys.exit('\n\nMust input mode\n\n')

    mode = args[1]
    modes = ['train', 'decode', 'mutrate']
    
    if mode not in modes:
        sys.exit('\n\nERROR! Mode must be either "mutrate", "train", or "decode"\n\n')
        
    elif mode == 'mutrate':
        # python3 main.py mutrate obs.bed 

        if len(args) < 2:
            sys.exit('\n\nMust input:\n\tObservations\n\n')
        obs = args[2]
        out = 'mutrates.txt'
        window_size = 1_000_000
        
        make_mutation_rate(obs, out, window_size)

    
    elif mode == 'train':
        # python3 main.py train obs.bed mutrates.txt

        if len(args) < 4:
            sys.exit('\n\nMust input:\n\tObservations\n\tMutation rate\n\n')
        obs, mutrates = args[2], args[3]
        out = 'trained.json'
        
        train(obs, mutrates, out)
    
    elif mode == 'decode':
        # python3 main.py decode obs.bed mutrates.txt trained.json
        
        if len(args) < 5:
            sys.exit('\n\nMust input:\n\tParameters\n\tObservations\n\tMutation rate\n\tOutput\n\n')
        
        obs, mutrates, params = args[2], args[3], args[4]
        out = 'probs_and_path.tsv'
        
        decode(obs, mutrates, params, out)


if __name__ == "__main__":
    main()
