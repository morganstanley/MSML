import pickle
import yaml
import numpy as np
from utils_gen import format_directory

def analyze_generative(experiment, yaml_file, runs=5, by_time=False, return_GIR = False):
    result_path = []
    for run in range(runs):
        yaml_filepath = experiment + yaml_file
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, net_savepath = format_directory(cfg, experiment, run)
        result_path.append(savepath)
        
    ENERGY = []
    
    for run, result in enumerate(result_path):
        stats_test = open(result + '/test_stats.pkl', 'rb')
        stats_dict_test = pickle.load(stats_test)
        ENERGY.append(stats_dict_test[0])
    print(stats_dict_test[1])
    return np.array(ENERGY)

experiment = ["power/", "miniboone/", "hepmass/", 
              "gas/", "cortex/"]
methods     =["wgan", "vae",  "maf"]
method_name = ["WGAN", "VAE", "MAF"]
datatype = ["Power", "Miniboone", "Hepmass", "Gas", "Cortex"]
samples = dict()
for i, e in enumerate(experiment):
    print(e)
    samples[datatype[i]] = {"method":[], "data":[]}
    for j, met in enumerate(methods):
        sampled = analyze_generative("{}".format(e), '{}.yaml'.format(met), by_time=False, runs=10, return_GIR=False)
        samp_flat = sampled.flatten().tolist()
        samples[datatype[i]]["data"] += samp_flat
        samples[datatype[i]]["method"] += [method_name[j]]*len(samp_flat)

for i,d in enumerate(datatype):
    for j, m in enumerate(method_name):
        print(np.round(np.mean(np.array(samples[datatype[i]]['data']).reshape(3,2), 1)[j], 3),
              "(" + str(np.round(np.std(np.array(samples[datatype[i]]['data']).reshape(3,2), 1)[j], 3))+ ")")
    print("")




