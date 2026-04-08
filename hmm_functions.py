import json
import numpy as np
from numba import njit
from math import lgamma
from scipy.optimize import minimize



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM Parameter Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions, dispersions): 
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)
        self.dispersions = np.array(dispersions)

    def __str__(self):
        out = f'> state_names = {self.state_names.tolist()}\n'
        out += f'> starting_probabilities = {np.matrix.round(self.starting_probabilities, 3).tolist()}\n'
        out += f'> transitions = {np.matrix.round(self.transitions, 3).tolist()}\n'
        out += f'> emissions = {np.matrix.round(self.emissions, 3).tolist()}\n'
        out += f'> dispersions = {np.matrix.round(self.dispersions, 3).tolist()}'
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state_names}, {self.starting_probabilities}, {self.transitions}, {self.emissions}, {self.dispersions})'


def read_HMM_parameters_from_file(filename):
    if filename is None:
        return get_default_HMM_parameters()

    with open(filename) as json_file:
        data = json.load(json_file)

    return HMMParam(state_names = data['state_names'], 
                    starting_probabilities = data['starting_probabilities'], 
                    transitions = data['transitions'], 
                    emissions = data['emissions'],
                    dispersions = data['dispersions'])


def get_default_HMM_parameters():
    return HMMParam(state_names = ['Human', 'Archaic'], 
                    starting_probabilities = [0.95, 0.05], 
                    transitions = [[0.99,0.01],[0.05,0.95]], 
                    emissions = [0.3, 4.2],
                    dispersions = [0.002, 0.144])


def write_HMM_to_file(hmmparam, outfile):
    data = {key: value.tolist() for key, value in vars(hmmparam).items()}
    json_string = json.dumps(data, indent = 2) 
    with open(outfile, 'w') as out:
        out.write(json_string)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit
def neg_binom_probability(k, lam, r):
    '''Calculate the probability of observing k given a negative binomial distribution with expectation lam and dispersion r'''
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # Need to change this if and when we change how the mutation rate is calculated
    
    p = r / (r + lam)
    
    # Work in log space to avoid underflow/overflow issues
    log_neg_binom = (lgamma(r + k) - lgamma(r) - lgamma(k + 1)
              + r * np.log(p)
              + k * np.log(1 - p))
    
    return np.exp(log_neg_binom)


@njit
def emission_probailities(observations, obs_rates, emissions, dispersions):
    '''Calculate the emission probabilities for each observation and state'''
    num_obs = len(observations)
    num_states = len(emissions)          
    probabilities = np.zeros((num_obs, num_states)) 
    
    for state in range(num_states):
        for t in range(num_obs):
            lam = emissions[state] * obs_rates[t]
            probabilities[t, state] = neg_binom_probability(observations[t], lam, dispersions[state])
    
    return probabilities


@njit
def fwd_step(alpha_prev, emission_prob, trans_mat):
    '''Calculate the forward probabilities for the next time step'''
    alpha_new = (alpha_prev @ trans_mat) * emission_prob
    n = np.sum(alpha_new)
    return alpha_new / n, n


@njit
def forward(emissions_probs, transitions, init_start):
    '''Calculate the forward probabilities for all time steps'''
    num_obs = len(emissions_probs)
    forwards_in = np.zeros( (num_obs, len(init_start)) ) 
    scales = np.ones(num_obs)

    for t in range(num_obs):
        if t == 0:
            forwards_in[t,:] = init_start  * emissions_probs[t,:]
            scales[t] = np.sum(forwards_in[t,:])
            forwards_in[t,:] = forwards_in[t,:] / scales[t]  
        else:
            forwards_in[t,:], scales[t] = fwd_step(forwards_in[t-1,:], emissions_probs[t,:], transitions) 

    return forwards_in, scales


@njit
def bwd_step(beta_next, emission_prob, trans_mat, n):
    '''Calculate the backward probabilities for the previous time step'''
    beta = (trans_mat * emission_prob) @ beta_next
    return beta / n


@njit
def backward(emissions_probs, transitions, scales):
    '''Calculate the backward probabilities for all time steps'''
    num_obs, num_states = emissions_probs.shape
    beta = np.ones((num_obs, num_states))
    for i in range(num_obs - 1, 0, -1):
        beta[i - 1,:] = bwd_step(beta[i,:], emissions_probs[i,:], transitions, scales[i])
    return beta


@njit
def get_log_likelihood(hmm_parameters, observations, obs_rates):
    '''Calculate the log-likelihood of the data given the HMM parameters'''
    emissions_probs = emission_probailities(observations, obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)
    _, scales = forward(emissions_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    return np.sum(np.log(scales))


@njit
def fwd_step_keep_track(alpha_prev, E, trans_mat):
    '''Calculate the forward probabilities for the next time step, while keeping track of which previous state contributed most to each current state'''
    
    # scaling factor
    n = np.sum((alpha_prev @ trans_mat) * E)
    
    results = np.zeros(len(E))
    back_track_states = np.zeros(len(E))

    for current_s in range(len(E)):
        for prev_s in range(len(E)):
            new_prob = alpha_prev[prev_s] * trans_mat[prev_s, current_s] * E[current_s] / n

            if new_prob > results[current_s]:
                results[current_s] = new_prob
                back_track_states[current_s] = prev_s

    return results, back_track_states


@njit
def viterbi(probabilities, transitions, init_start):
    '''Calculate the Viterbi path and its probability for all time steps'''
    num_obs = len(probabilities)
    forwards_in = np.zeros((num_obs, len(init_start))) 
    backtracks = np.zeros((num_obs, len(init_start)), dtype=np.int32) 

    for t in range(num_obs):
        if t == 0:
            forwards_in[t,:] = init_start * probabilities[t,:]
            scale_param = np.sum( forwards_in[t,:] )
            forwards_in[t,:] = forwards_in[t,:] / scale_param
        else:
            forwards_in[t,:], backtracks[t,:] = fwd_step_keep_track(forwards_in[t-1,:], probabilities[t,:], transitions) 

    return forwards_in, backtracks


@njit
def nb_neg_log_likelihood(params, gamma_s, obs, mutrates):
    '''Calculate negative log-likelihood of the data given the parameters for a single state, weighted by the posterior probability of being in that state'''
    e_s = np.exp(params[0])
    r_s = np.exp(params[1])

    mu = e_s * mutrates
    p  = r_s / (r_s + mu)

    log_nb = (lgamma(r_s + obs) - lgamma(r_s) - lgamma(obs + 1)
              + r_s * np.log(p)
              + obs * np.log(np.maximum(1 - p, 1e-300)))

    return -np.sum(gamma_s * log_nb)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def logoutput(hmm_parameters, loglikelihood, iteration):

    n_states = len(hmm_parameters.emissions)

    # Make header
    if iteration == 0:    
        print_emissions = '\t'.join(['emis{0}'.format(x + 1) for x in range(n_states)])
        print_starting_probabilities = '\t'.join(['start{0}'.format(x + 1) for x in range(n_states)])
        print_transitions = '\t'.join(['trans{0}_{0}'.format(x + 1) for x in range(n_states)])
        print_dispersions = '\t'.join(['disp{0}'.format(x + 1) for x in range(n_states)])
        print(f'iter\tlog-l\t{print_starting_probabilities}\t{print_emissions}\t{print_transitions}\t{print_dispersions}')

    # Print parameters
    print_emissions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.emissions, 4)])
    print_starting_probabilities = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.starting_probabilities, 3)])
    print_transitions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.transitions, 4).diagonal()])
    print_dispersions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.dispersions, 4)])
    print(f'{iteration}\t{round(loglikelihood, 4)}\t{print_starting_probabilities}\t{print_emissions}\t{print_transitions}\t{print_dispersions}')


def maximize_emissions_dispersions(posterior_probs, observations, obs_rates, current_emissions, current_dispersions):
    num_states = posterior_probs.shape[1]
    new_emissions = np.zeros(num_states)
    new_dispersions = np.zeros(num_states)

    for state in range(num_states):
        gamma_s = posterior_probs[:, state]
        
        start_params = [np.log(current_emissions[state]), np.log(current_dispersions[state]),]
        
        result = minimize(nb_neg_log_likelihood, start_params, args=(gamma_s, observations, obs_rates), method='L-BFGS-B')
        
        new_emissions[state] = np.exp(result.x[0])
        new_dispersions[state] = np.exp(result.x[1])

    return new_emissions, new_dispersions


def TrainBaumWelsch(hmm_parameters, observations, obs_rates):

    num_states = len(hmm_parameters.starting_probabilities)

    emissions_probs = emission_probailities(observations, obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)
    forward_probs, scales = forward(emissions_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    backward_probs = backward(emissions_probs, hmm_parameters.transitions, scales)

    # Update starting probs
    posterior_probs = forward_probs * backward_probs
    scale = np.sum(posterior_probs)
    new_starting_probabilities = np.sum(posterior_probs, axis=0) / scale

    # Update emissions and dispersions
    new_emissions, new_dispersions = maximize_emissions_dispersions(posterior_probs, observations, obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)

    # Update transition probs
    new_transitions_matrix = np.zeros((num_states, num_states))
    for state1 in range(num_states):
        for state2 in range(num_states):
            new_transitions_matrix[state1, state2] = np.sum(
                forward_probs[:-1, state1] * backward_probs[1:, state2]
                * hmm_parameters.transitions[state1, state2]
                * emissions_probs[1:, state2] / scales[1:]
            )
    new_transitions_matrix /= new_transitions_matrix.sum(axis=1)[:, np.newaxis]

    return HMMParam(hmm_parameters.state_names, new_starting_probabilities, new_transitions_matrix, new_emissions, new_dispersions)


def TrainModel(observations, obs_rates, hmm_parameters, epsilon = 1e-3, maxiterations = 1000):
    
    # Get probability of data with initial parameters
    previous_loglikelihood = get_log_likelihood(hmm_parameters, observations, obs_rates)
    logoutput(hmm_parameters, previous_loglikelihood, 0)
    
    # Train parameters using Baum-Welch algorithm
    for iter in range(1, maxiterations):
        hmm_parameters = TrainBaumWelsch(hmm_parameters, observations, obs_rates) # Maximization
        new_loglikelihood = get_log_likelihood(hmm_parameters, observations, obs_rates) # Expectation
        
        logoutput(hmm_parameters, new_loglikelihood, iter)

        if abs(new_loglikelihood - previous_loglikelihood) < epsilon:
            break 

        previous_loglikelihood = new_loglikelihood

    return hmm_parameters


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Decode (posterior decoding)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def Calculate_Posterior_probabillities(emissions, hmm_parameters):
    """Get posterior probability of being in state s at time t"""
    
    forward_probs, scales = forward(emissions, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    backward_probs = backward(emissions, hmm_parameters.transitions, scales)
    posterior_probabilities = (forward_probs * backward_probs).T

    return posterior_probabilities


def PMAP_path(posterior_probabilities):
    """Get maximum posterior decoding path"""
    path = np.argmax(posterior_probabilities, axis = 0)
    return path 


def Viterbi_path(emissions, hmm_parameters):
    """Get Viterbi path (most likely path)"""
    n_obs, _ = emissions.shape
    
    viterbi_probs, backtracks = viterbi(emissions, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    
    # backtracking
    viterbi_path = np.zeros(n_obs, dtype = int)
    viterbi_path[-1] = np.argmax(viterbi_probs[-1,:])
    for t in range(n_obs - 2, -1, -1):
        viterbi_path[t] = backtracks[t + 1, viterbi_path[t + 1]]

    return viterbi_path

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Write segments to output file
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def Write_posterior_probs(obs, mutrates, post_seq, post_path, viterbi_path, hmm_parameters, filename, window_size):
    post_seq = post_seq.T

    with open(filename, 'w') as out:
        state_names = '\t'.join(hmm_parameters.state_names)
        out.write('start\tend\tnum_kmers\tmutationrate\t' + state_names + '\tposterior_state\tviterbi_state\n')

        i = 0
        for (obs, m, posterior, post_state, viterbi_state) in zip(obs, mutrates, post_seq, post_path, viterbi_path):
            posterior_to_print = '\t'.join([str(round(x, 4)) for x in posterior])
            out.write(f'{i * window_size}\t{(i + 1) * window_size}\t{obs}\t{m}\t{posterior_to_print}\t{hmm_parameters.state_names[post_state]}\t{hmm_parameters.state_names[viterbi_state]}\n')
            i += 1
