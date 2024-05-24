from sim_process import *
from scipy.io import loadmat
from utils import *
import numpy as np
import glob
import pandas as pd
import os


def concat_chemo_traj(data, min_timestep=100):
    """
    Concatentate data and filter short chemotaxi trajectories
    """
    flatten_data = np.zeros((data.shape[0], *(min_timestep,10)))
    nonzeros = []
    for i in range(data.shape[0]):
        try:
            flatten_data[i] = data[i][0:min_timestep,:]
            nonzeros.append(i)
        except:
            continue
    flatten_data = flatten_data[nonzeros, :, 1:4]
    return flatten_data

def read_EEG(path, dataset_type):
    """
    Read chemo-taxi helper
    """
    eeg_path = glob.glob(path + dataset_type + ("" if '.rd' in dataset_type else "/*"))
    
    eeg_data = []
    for data in eeg_path:
        eeg_temp = []
        S2 = False
        with open(data) as f:
            for i, line in enumerate(f):
                if i < 4:
                    if 'S2' in line:
                        S2 = True
                        break
                    continue
                if " chan " in line:
                    temp = []
                    eeg_temp.append(temp)
                else:
                    temp.append(line.split(" ")[-1])
        if not S2:
            eeg_data.append(eeg_temp)
    eeg = np.array(eeg_data).astype(float)
    eeg = eeg.reshape(*(eeg.shape),1)
    if eeg.shape[0] == 1:
        eeg = eeg.squeeze(0)
    return eeg

def get_none_simu(path, 
                  dataset_type,
                  subset_time=None, 
                  min_timestep=100, 
                  split_type="timestep", 
                  split_size=None,
                  seed = 0,
                  **params
                 ):
    
    """
    Retrieve Real data, and split them base on split_type and split_size
    """    
    N_train = params.setdefault('N_train', 100)
    N_val   = params.setdefault('N_val', 500)
    N_test  = params.setdefault('N_test', 500)
    
    xs_train = xs_val = xs_test = None
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if "EEG" in path:
        data = read_EEG(path, dataset_type)
    elif "chemotaxi" in path:
        data = loadmat(path)[dataset_type][0][0][0].flatten()
        data = concat_chemo_traj(data, min_timestep)
    elif "realGen" in path:
        # Adapted from https://github.com/dargilboa/mdma/tree/8ed4afc731936f695d6e320d804e5157e8eb8a71/experiments/UCI
        tot_N = N_train + N_val + N_test
        if 'Cortex' in dataset_type:
            data = pd.read_csv(path + dataset_type).values[:,2:78].astype(float)
            data = np.nan_to_num(data)
        elif 'miniboone' in dataset_type:
            data = np.load(path+dataset_type)
        elif 'ethylene_CO' in dataset_type:
            data = pd.read_pickle(path + dataset_type)
            data.drop("Meth", axis=1, inplace=True)
            data.drop("Eth", axis=1, inplace=True)
            data.drop("Time", axis=1, inplace=True)
            C = data.corr()
            A = C > 0.98
            B = A.values.sum(axis=1)
            while np.any(B > 1):
                col_to_remove = np.where(B > 1)[0][0]
                col_name = data.columns[col_to_remove]
                data.drop(col_name, axis=1, inplace=True)
                C = data.corr()
                A = C > 0.98
                B = A.values.sum(axis=1)
            data = (data - data.mean()) / data.std()
            data = data.values.astype(float)
        elif 'power' in dataset_type:
            data = np.load(path+dataset_type)
            N = data.shape[0]
            data = np.delete(data, 3, axis=1)
            data = np.delete(data, 1, axis=1)
            voltage_noise = 0.01 * np.random.rand(N, 1)
            gap_noise = 0.001 * np.random.rand(N, 1)
            sm_noise = np.random.rand(N, 3)
            time_noise = np.zeros((N, 1))
            noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
            data = data + noise
            data = data.astype(float)
        elif 'hepmass' in dataset_type:
            data = pd.read_csv(filepath_or_buffer=os.path.join(path,dataset_type),
                           index_col=False)
            data = data[data[data.columns[0]] == 1]
            data = data.drop(data.columns[0], axis=1)
            data = data.values.astype(float)
            
        # subsampling
        index = np.random.choice(list(range(data.shape[0])), np.min([tot_N, data.shape[0]]), False)
        data = data[index]
        
    order = np.random.choice(range(len(data)), len(data), False)
    data = data[order, ...]
    
    if subset_time:
        data = data[...,0:subset_time,:]
    tot_T = data.shape[-2]
    tot_N = data.shape[0]
    train_split_t = None
    train_split_n = None
    if split_size:
        splits = np.array(split_size.split("-")).astype(float)
        if split_type == "data":
            train_split = int(np.round(splits[0]*tot_N, 0))
            val_split   = train_split + int(np.round(splits[1]*tot_N, 0))
            test_split  = tot_N
            
            xs_train = data[0:train_split, ...]
            ts_train = np.array(range(tot_T))
            
            xs_val = data[train_split:val_split, ...]
            ts_val = ts_train
            
            xs_test = data[val_split:tot_N, ...]
            ts_test = ts_train
                
        elif split_type == "timestep":
            train_split = int(np.round(splits[0]*tot_T, 0))
            val_split   = train_split + int(np.round(splits[1]*tot_T, 0))
            test_split  = tot_T
            xs_train = data[...,0:train_split, :]
            ts_train = np.array(range(train_split))
            
            xs_val = data[...,train_split:val_split, :]
            ts_val = np.array(range(train_split, val_split))
            
            xs_test = data[...,val_split:test_split, :]
            ts_test = np.array(range(val_split, test_split))
            
        elif split_type == "both":
            train_split_n = int(np.round(splits[0]*tot_N, 0))
            val_split_n   = train_split_n + int(np.round(splits[1]*tot_N, 0))
            test_split_n  = tot_N
            
            train_split_t = int(np.round(splits[3]*tot_T, 0))
            
            xs_train = data[0:train_split_n,...,0:train_split_t, :]
            ts_train = np.array(range(train_split_t))
            
            xs_val = data[train_split_n:val_split_n,..., :]
            ts_val = np.array(range(tot_T))
            
            xs_test = data[val_split_n:,...,:]
            ts_test = np.array(range(tot_T))
            
    else:
        # special case for real data in generative modeling
        data_train = data[0:N_train]
        data_validate = data[N_train:N_train + N_val]
        if "hepmass" in dataset_type:
            data_test = pd.read_csv(filepath_or_buffer=os.path.join(path,"hepmass_1000_test.csv"),
                                        index_col=False)
            data_test = data_test[data_test[data_test.columns[0]] == 1]
            data_test = data_test.drop(data_test.columns[0], axis=1)
            # Because the data set is messed up!
            data_test = data_test.drop(data_test.columns[-1], axis=1)
            data_test = data_test.values.astype(float)
            np.random.shuffle(data_test)
            data_test = data_test[:N_test]
        else:
            data_test = data[N_train + N_val:np.min([data.shape[0], N_train + N_val + N_test])]

        # normalize
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s
        
        # reshape to 3-d
        xs_train = np.expand_dims(data_train,axis=1)
        ts_train = np.array(range(2))
        xs_val = np.expand_dims(data_validate,axis=1)
        ts_val = np.array(range(2))
        xs_test = np.expand_dims(data_test,axis=1)
        ts_test = np.array(range(2))
        
        
    return xs_train, ts_train, xs_val, ts_val, xs_test, ts_test, train_split_t
            

def get_data(dataset_params, simulation, extrapolate, generative, add_noise = False, noise_level=0.5, FK=False, **params):
    """
    Get data wrapper
    """
    xs_train = ts_train = xs_val = ts_val = xs_test = ts_test = None  
    if simulation:
        if generative:
            if "path" in set(list(dataset_params.keys())) and "realGen" in dataset_params["path"]:
                train_data, _, val_data, _, test_data, _, _ = get_none_simu(**dataset_params)
                train_data = train_data.squeeze(1)
                val_data = val_data.squeeze(1)
                test_data = test_data.squeeze(1)
                xs_train, ts_train, _, irreg_t = BB_to_data(**dataset_params, min_dist=False, input_data = train_data)
                dataset_params['n_bridge'] = 1
                xs_val, ts_val, _, irreg_t = BB_to_data(**dataset_params, min_dist=True, input_data = val_data)
                dataset_params['n_bridge'] = 1
                xs_test, ts_test, _, irreg_t = BB_to_data(**dataset_params, min_dist=True, input_data = test_data)
                
            elif "path" not in set(list(dataset_params.keys())): 
                n_particles_test = params.setdefault('n_particles_test', 100)
                n_particle_train = dataset_params['n_samples']
                
                dataset_params['n_samples'] = n_particle_train
                xs_train, ts_train, _, irreg_t, = BB_to_data(**dataset_params, min_dist=False)
                
                dataset_params['n_samples'] = n_particles_test
                dataset_params['n_bridge'] = 1
                xs_val, ts_val, _, irreg_t, = BB_to_data(**dataset_params, min_dist=True)
                
                dataset_params['n_samples'] = n_particles_test
                dataset_params['n_bridge'] = 1
                dataset_params['seed']      = 100 - dataset_params['seed']
                xs_test, ts_test, _, irreg_t, = BB_to_data(**dataset_params, min_dist=True)
                
            if FK:
                xs_train = xs_train[:,-1,:].reshape(xs_train.shape[0], xs_train.shape[-1])
                xs_val = xs_val[:,-1,:].reshape(xs_val.shape[0], xs_val.shape[-1])
                xs_test = xs_test[:,-1,:].reshape(xs_test.shape[0], xs_test.shape[-1])
        elif extrapolate:
            xs_train, ts_train, _, irreg_t = sim_process_mckean(**dataset_params)
        else:
            try:
                atlas = dataset_params['atlas']
            except KeyError:
                atlas = False
                
            try:
                intfire = dataset_params['intfire']
            except KeyError:
                intfire = False
            
            n_particles_test = params.setdefault('n_particles_test', 100)
            n_particle_train = dataset_params['n_particles']
            dataset_params['n_particles'] = n_particle_train + n_particles_test + n_particles_test
            if atlas:
                data, ts_train, _, irreg_t = sim_process_meanfield_atlas(**dataset_params)
            elif intfire:
                data, ts_train, _, irreg_t = sim_process_meanfield_intfire(**dataset_params)
            else:
                data, ts_train, _, irreg_t = sim_process_mckean(**dataset_params)
            
            order = np.random.choice(range(len(data)), len(data), False)
            data = data[order, ...]
            
            train_split = n_particle_train
            val_split   = train_split + n_particles_test
            test_split  = dataset_params['n_particles']

            xs_train = data[0:train_split, ...]
            if add_noise:
                xs_train += np.random.normal(0, noise_level, xs_train.shape)
                
            xs_val = data[train_split:val_split, ...]
            ts_val = ts_train

            xs_test = data[val_split:, ...]
            ts_test = ts_train
        return xs_train, ts_train, xs_val, ts_val, xs_test, ts_test, None, irreg_t
    else:
        xs_train, ts_train, xs_val, ts_val, xs_test, ts_test, train_split_t = get_none_simu(**dataset_params)
        return xs_train, ts_train, xs_val, ts_val, xs_test, ts_test, train_split_t, None
    
def load_data(cfg, extrapolate=False, generative=False, add_noise=False, noise_level=0.5, FK=False):
    dataset_params = cfg['dataset']
    try:
        scale_t = dataset_params["scale_t"]
    except:
        scale_t = None
        
    labeling = 0
    try:
        deepAR_param = None
        if cfg["deepAR"] is not None:
            deepAR_param = cfg['deepAR']
        elif cfg["transformer"] is not None:
            deepAR_param = cfg['transformer']
        NF_params    = None
        MLP_params   = None
        driftMLP_param = None
    except KeyError:
        deepAR_param = None
        NF_params      = cfg['NF']
        MLP_params     = (cfg['MF'] if cfg['MF'] else cfg['MLP'])
        driftMLP_param = MLP_params['net']
        
        try: 
            labeling = driftMLP_param['label_x']
        except KeyError:
            labeling = 0
        
    if deepAR_param is not None:
        window_size = deepAR_param["window_size"]

    
    loader_params  = cfg['loader']
    batch_size_paths = cfg['loader']['batch_size_paths']
    batch_size_points = (cfg['loader']['batch_size_points'] if NF_params else None)
    
    point_loader      = None
    train_path_loader = None
    val_path_loader   = None
    test_path_loader  = None
    
    data = get_data(dataset_params, 
                    dataset_params["simulation"],
                    extrapolate=extrapolate,
                    generative=generative,
                    n_particles_test=cfg['other']['test_n_particles'],
                    add_noise = add_noise, 
                    noise_level=noise_level,
                    FK=FK)
    
    xs_train, ts_train, xs_val, ts_val, xs_test, ts_test, train_split_t, irreg_t = data
    
    dt = ts_train[1] - ts_train[0]
    
    try:
        partition = dataset_params["partition"]
    except KeyError:
        partition = None
    
    if len(xs_train.shape) < 4:
        if deepAR_param is not None:
            # Pad Data from benchmark deepAR
            path_tensor, path_labels = pad_data(xs_train, window_size)
            # Adjust batch size for equal number of optimization steps
            batch_size_paths = int((path_tensor.shape[0]/xs_train.shape[0])*batch_size_paths)
        else:
            path_tensor = torch.from_numpy(xs_train)
            path_labels = torch.from_numpy(ts_train).repeat(xs_train.shape[0],1)*(scale_t if scale_t else 1)
            if FK:
                path_labels = path_labels[:,-1].reshape(path_labels.shape[0])
        
        if labeling != 0:
            particle_labels = make_particle_label(x0 = xs_train[:,0,:], partition = partition)
            train_path_dataset = torch.utils.data.TensorDataset(path_tensor, 
                                                                path_labels, 
                                                                particle_labels)
        else: 
            train_path_dataset = torch.utils.data.TensorDataset(path_tensor, path_labels)
            if FK:
                train_path_dataset = torch.utils.data.TensorDataset(path_tensor)
            
        if deepAR_param is not None:
            train_path_loader = torch.utils.data.DataLoader(train_path_dataset, batch_size=batch_size_paths, shuffle=True)
        else: 
            train_path_loader = torch.utils.data.DataLoader(train_path_dataset, batch_size=batch_size_paths, shuffle=True)
        
        if xs_val is not None:
            if deepAR_param is not None:
                path_tensor_val, path_labels_val = pad_data(xs_val, window_size)
                
            else:
                path_tensor_val = torch.from_numpy(xs_val)
                path_labels_val = torch.from_numpy(ts_val).repeat(xs_val.shape[0],1)*(scale_t if scale_t else 1)
            if labeling != 0:
                particle_labels_val = make_particle_label(x0 = xs_val[:,0,:], partition = partition)
                val_path_dataset = torch.utils.data.TensorDataset(path_tensor_val, 
                                                                  path_labels_val, 
                                                                  particle_labels_val)
            else: 
                val_path_dataset = torch.utils.data.TensorDataset(path_tensor_val, path_labels_val)
                if FK:
                    val_path_dataset = torch.utils.data.TensorDataset(path_tensor_val)
            if deepAR_param:
                val_path_loader = torch.utils.data.DataLoader(val_path_dataset, batch_size=xs_val.shape[0], shuffle=False)
            else: 
                val_path_loader = torch.utils.data.DataLoader(val_path_dataset, batch_size=batch_size_paths, shuffle=False)
            
        if xs_test is not None:
            if deepAR_param is not None:
                path_tensor_test, path_labels_test = pad_data(xs_test, window_size)
            else:
                path_tensor_test = torch.from_numpy(xs_test)
                path_labels_test = torch.from_numpy(ts_test).repeat(xs_test.shape[0],1)*(scale_t if scale_t else 1)
                if FK:
                    path_labels_test = path_labels_test[:,-1].reshape(path_labels_test.shape[0])
            if labeling != 0:
                particle_labels_test = make_particle_label(x0 = xs_test[:,0,:], partition = partition)
                test_path_dataset = torch.utils.data.TensorDataset(path_tensor_test, 
                                                                   path_labels_test, 
                                                                   particle_labels_test)
            else:
                test_path_dataset = torch.utils.data.TensorDataset(path_tensor_test, path_labels_test)
                if FK:
                    test_path_dataset = torch.utils.data.TensorDataset(path_tensor_test)
            if deepAR_param:
                test_path_loader = torch.utils.data.DataLoader(test_path_dataset, batch_size=xs_test.shape[0], shuffle=False)
            else: 
                test_path_loader = torch.utils.data.DataLoader(test_path_dataset, batch_size=batch_size_paths, shuffle=False)
            
            
        if NF_params is not None:
            point_tensor = torch.from_numpy(xs_train.reshape(xs_train.shape[0]*xs_train.shape[1], -1))
            point_labels = torch.from_numpy(ts_train).repeat(xs_train.shape[0],1).reshape(xs_train.shape[0]*ts_train.shape[0],-1)
            print(point_labels.shape)
            point_dataset = torch.utils.data.TensorDataset(point_tensor, point_labels)
            point_loader = torch.utils.data.DataLoader(point_dataset, batch_size=batch_size_points, shuffle=True)
            
        if deepAR_param is not None:
            return train_path_loader, val_path_loader, test_path_loader, dt, train_split_t, xs_val, xs_test

        return point_loader, train_path_loader, val_path_loader, test_path_loader, dt, train_split_t, irreg_t