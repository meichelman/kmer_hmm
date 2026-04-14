# Pipeline that performs the following:
# 1. Makes the observation rate
# 2. Trains the HMM
# 3. Decodes the observations


configfile: "config.yaml"


# Configuration variables
obs_file = config["observations"]
obs_rate_out = config["observation_rate"]["output_file"]
obs_rate_bin_size = config["observation_rate"]["bin_size"]
# train_start_params = config["train"]["start_params"]
train_out = config["train"]["output_file"]
use_viterbi = config["decode"]["use_viterbi"]
decode_out_path = config["decode"]["output_path_file"]
decode_out_tracts = config["decode"]["output_tracts_file"]


rule all:
    input:
        decode_out_path,
        decode_out_tracts
    localrule: True


rule decode:
    input:
        obs_file,
        obs_rate_out,
        train_out
    output:
        decode_out_path,
        decode_out_tracts
    params:
        use_viterbi=use_viterbi
    localrule: True
    run:
        if params.use_viterbi:
            szCommand = f"python decode.py -obs {input[0]} -obs_rates {input[1]} -param {input[2]} -viterbi {params.use_viterbi} -out_path {output[0]} -out_tracts {output[1]}"
        else:
            szCommand = f"python decode.py -obs {input[0]} -obs_rates {input[1]} -param {input[2]} -out_path {output[0]} -out_tracts {output[1]}"
        shell(szCommand)


rule train:
    input:
        obs_file,
        obs_rate_out
    output:
        train_out
    localrule: True
    run:
        szCommand = f"python train.py -obs {input[0]} -obs_rates {input[1]} -out {output}"
        shell(szCommand)


rule make_observation_rate:
    input:
        obs_file
    output:
        obs_rate_out
    params:
        bin_size=obs_rate_bin_size
    localrule: True
    run:
        szCommand = f"python observation_rate.py -obs {input} -out {output} -bin_size {params.bin_size}"
        shell(szCommand)
