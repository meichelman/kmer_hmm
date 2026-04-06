import numpy as np
from numba import njit
import json
import math

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM Parameter Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions): 
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)

    def __str__(self):
        out = f'> state_names = {self.state_names.tolist()}\n'
        out += f'> starting_probabilities = {np.matrix.round(self.starting_probabilities, 3).tolist()}\n'
        out += f'> transitions = {np.matrix.round(self.transitions, 3).tolist()}\n'
        out += f'> emissions = {np.matrix.round(self.emissions, 3).tolist()}'
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state_names}, {self.starting_probabilities}, {self.transitions}, {self.emissions})'


# Read HMM parameters from a json file
def read_HMM_parameters_from_file(filename):

    if filename is None:
        return get_default_HMM_parameters()

    with open(filename) as json_file:
        data = json.load(json_file)

    return HMMParam(state_names = data['state_names'], 
                    starting_probabilities = data['starting_probabilities'], 
                    transitions = data['transitions'], 
                    emissions = data['emissions'])


# Set default parameters
def get_default_HMM_parameters():
    return HMMParam(state_names = ['Human', 'Archaic'], 
                    starting_probabilities = [0.5, 0.5], 
                    transitions = [[0.5,0.5],[0.5,0.5]], 
                    emissions = [5, 5])


# Save HMMParam to a json file
def write_HMM_to_file(hmmparam, outfile):
    data = {key: value.tolist() for key, value in vars(hmmparam).items()}
    json_string = json.dumps(data, indent = 2) 
    with open(outfile, 'w') as out:
        out.write(json_string)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit
def poisson_probability_underflow_safe(n, lam):
    # naive:   np.exp(-lam) * lam**n / factorial(n)
    
    # iterative, to keep the components from getting too large or small:
    p = np.exp(-lam)
    for i in range(n):
        p *= lam
        p /= i+1
    return p


@njit
def NB_probability_underflow_safe(k, r, p):
    # Compute log P(X=n) to avoid underflow entirely
    q = 1.0 - p
    log_p = math.log(p)
    log_q = math.log(q)

    # log comb(n+r-1, n) computed iteratively
    log_comb = 0.0
    for i in range(k):
        log_comb += math.log(r + i) - math.log(i + 1)

    return math.exp(log_comb + k * log_q + r * log_p)


@njit
def Emission_probs(emissions, observations, mutrates, window_size):
    n = len(observations)
    n_states = len(emissions)          
    
    probabilities = np.zeros( (n, n_states) ) 
    # for state in range(n_states): 
    #     for index in range(n):
    #         # lam = emissions[state] * mutrates[index]
    #         # probabilities[index,state] = poisson_probability_underflow_safe(observations[index], lam)
    #         p = emissions[state] * observations[index] / window_size
    #         k = window_size - 1 - observations[index]
    #         if p > 0:
    #             probabilities[index,state] = NB_probability_underflow_safe(k, observations[index], p)
    #         else:
    #             probabilities[index,state] = 0.0
    arc_state = 1
    hum_state = 0
    for index in range(n):
        p = emissions[arc_state] * mutrates
        k = window_size - 1 - observations[index]
        probabilities[index,arc_state] = NB_probability_underflow_safe(k, observations[index], p)
        probabilities[index,hum_state] = 1 - probabilities[index,arc_state]
            
    probabilities = np.where(probabilities < 1e-10, 1e-10, probabilities)
    return probabilities


@njit
def fwd_step(alpha_prev, E, trans_mat):
    alpha_new = (alpha_prev @ trans_mat) * E
    # looks like:
    # alpha_new = (0.5, 0.5) @ [[0.9999 0.0001] [0.02 0.98]] = (0.50995 0.49005) * ...
    n = np.sum(alpha_new)
    return alpha_new / n, n


@njit
def forward(probabilities, transitions, init_start):
    n = len(probabilities)
    forwards_in = np.zeros( (n, len(init_start)) ) 
    scale_param = np.ones(n)

    for t in range(n):
        if t == 0:
            forwards_in[t,:] = init_start  * probabilities[t,:]
            scale_param[t] = np.sum( forwards_in[t,:] )
            forwards_in[t,:] = forwards_in[t,:] / scale_param[t]  
        else:
            forwards_in[t,:], scale_param[t] = fwd_step(forwards_in[t-1,:], probabilities[t,:], transitions) 

    return forwards_in, scale_param


@njit
def bwd_step(beta_next, E, trans_mat, n):
    beta = (trans_mat * E) @ beta_next
    return beta / n


@njit
def backward(emissions, transitions, scales):
    n, n_states = emissions.shape
    beta = np.ones((n, n_states))
    for i in range(n - 1, 0, -1):
        beta[i - 1,:] = bwd_step(beta[i,:], emissions[i,:], transitions, scales[i])
    return beta


def GetProbability(hmm_parameters, obs, mutrates, window_size):

    emissions = Emission_probs(hmm_parameters.emissions, obs, mutrates, window_size)
    _, scales = forward(emissions, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    forward_probility_of_obs = np.sum(np.log(scales))

    return forward_probility_of_obs


@njit
def fwd_step_keep_track(alpha_prev, E, trans_mat):
    
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
    n = len(probabilities)
    forwards_in = np.zeros( (n, len(init_start)) ) 
    backtracks = np.zeros( (n, len(init_start)), dtype=np.int32 ) 

    for t in range(n):
        if t == 0:
            forwards_in[t,:] = init_start * probabilities[t,:]
            scale_param = np.sum( forwards_in[t,:] )
            forwards_in[t,:] = forwards_in[t,:] / scale_param
        else:
            forwards_in[t,:], backtracks[t,:] = fwd_step_keep_track(forwards_in[t-1,:], probabilities[t,:], transitions) 

    return forwards_in, backtracks


@njit
def calculate_log(x):
	return np.log(x)


@njit
def hybrid_step(prev, alpha, em, trans):
    value = prev + alpha * calculate_log(em * trans)
    best_state = np.argmax(value)
    max_prob = value[best_state]

    return best_state, max_prob


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
        print(f'iter\tlog-l\t{print_starting_probabilities}\t{print_emissions}\t{print_transitions}')

    # Print parameters
    print_emissions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.emissions, 4)])
    print_starting_probabilities = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.starting_probabilities, 3)])
    print_transitions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.transitions, 4).diagonal()])
    print(f'{iteration}\t{round(loglikelihood, 4)}\t{print_starting_probabilities}\t{print_emissions}\t{print_transitions}')


def TrainBaumWelsch(hmm_parameters, obs, mutrates, window_size):
    """
    Trains the model once, using the forward-backward algorithm. 
    """

    n_states = len(hmm_parameters.starting_probabilities)

    emissions = Emission_probs(hmm_parameters.emissions, obs, mutrates, window_size)
    print(f'emissions[:10]: {emissions[:10]}')
    forward_probs, scales = forward(emissions, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    print(f'forward_probs[:10]: {forward_probs[:10]}')
    backward_probs = backward(emissions, hmm_parameters.transitions, scales)
    # print(f'backward_probs[:10]: {backward_probs[:10]}')

    # Update starting probs
    posterior_probs = forward_probs * backward_probs
    print(f'posterior_probs[:10]: {posterior_probs[:10]}')
    normalize = np.sum(posterior_probs)
    new_starting_probabilities = np.sum(posterior_probs, axis=0)/normalize 

    # Update emission
    new_emissions_matrix = np.zeros((n_states))
    for state in range(n_states):
        top = np.sum(posterior_probs[:,state] * obs)
        # bottom = np.sum(posterior_probs[:,state] * mutrates)
        bottom = np.sum(posterior_probs[:,state])
        new_emissions_matrix[state] = top/bottom

    # Update Transition probs 
    new_transitions_matrix =  np.zeros((n_states, n_states))
    for state1 in range(n_states):
        for state2 in range(n_states):
            new_transitions_matrix[state1,state2] = np.sum( forward_probs[:-1,state1] * backward_probs[1:,state2] * hmm_parameters.transitions[state1, state2] * emissions[1:,state2]/ scales[1:] )
    new_transitions_matrix /= new_transitions_matrix.sum(axis=1)[:,np.newaxis]

    return HMMParam(hmm_parameters.state_names,new_starting_probabilities, new_transitions_matrix, new_emissions_matrix)


def TrainModel(obs, mutrates, hmm_parameters, window_size, epsilon = 1e-3, maxiterations = 1000):

    # Get probability of data with initial parameters
    previous_loglikelihood = GetProbability(hmm_parameters, obs, mutrates, window_size)
    logoutput(hmm_parameters, previous_loglikelihood, 0)
    
    # Train parameters using Baum Welch algorithm
    # for i in range(1,maxiterations):
    for i in range(2):
        hmm_parameters = TrainBaumWelsch(hmm_parameters, obs, mutrates, window_size)
        new_loglikelihood = GetProbability(hmm_parameters, obs, mutrates, window_size)
        logoutput(hmm_parameters, new_loglikelihood, i)

        if abs(new_loglikelihood - previous_loglikelihood) < epsilon:
            break 

        previous_loglikelihood = new_loglikelihood

    # Write the optimal parameters
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
            posterior_to_print = '\t'.join([str(round(x, 8)) for x in posterior])
            out.write(f'{i * window_size}\t{(i + 1) * window_size}\t{obs}\t{m}\t{posterior_to_print}\t{hmm_parameters.state_names[post_state]}\t{hmm_parameters.state_names[viterbi_state]}\n')
            i += 1
