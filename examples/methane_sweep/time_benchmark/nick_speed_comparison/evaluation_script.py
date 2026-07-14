from training_script import load_db, make_model
import hippynn
hippynn.settings.WARN_LOW_DISTANCES=False
from tqdm.auto import tqdm
import os
import time

from hippynn.graphs.gops import search_by_name

import torch



if __name__ == "__main__":

    n_reps_per_config = 5 # suggest 5 for production

    batch_size_list = [32,64,128,256]
    batch_size_list = [2048]
    
    config_list=[
        dict(tensor_model="HOP", tensor_order=ell, tensor_factors=en) for ell in [3,2,1] for en in [4,3,2] if not (ell==1 and en > 2)
        ] + \
        [
        dict(tensor_model="TS", tensor_order=ell, tensor_factors=0) for ell in [1,2]
        ] + \
        [dict(tensor_model="NONE", tensor_order=0, tensor_factors=0)]

    #config_list = config_list[-1:] # only test last config

# load database

    print("all configs",config_list)
    network_parameters = {
        "possible_species": [0, 1, 6, 7, 8],
        "n_features": 128,
        "n_sensitivities": 20,
        "dist_soft_min": 0.75,
        "dist_soft_max": 5.5,
        "dist_hard_max": 6.5,
        "n_interaction_layers": 2,
        "n_atom_layers": 3,
        }                                                                   

    use_ccx_subset = True
    en_name = "wb97x_dz.energy"
    force_name ="wb97x_dz.forces"
    db_info = dict(inputs=["atomic_numbers","coordinates"],targets=[en_name,force_name])
    seed = 0
    anidata_location = "/vast/home/nlubbers/hippynn_tests/release/datasets/ani1x_release/ani1x-release.h5"
    n_workers=0   
    db = load_db(db_info, en_name, force_name, seed, anidata_location, n_workers, use_ccx_subset)
    db.send_to_device("cuda:0")
    del db.splits['valid']
    del db.splits['train']

    # loop over configs
    all_times = {}

    for config in tqdm(config_list,desc='configs'):

        # build model
        henergy, force = make_model(network_parameters, **config, atomization_consistent=False,group_norm=True)
        
        species = search_by_name([henergy],"atomic_numbers")
        coords = search_by_name([henergy],"coordinates")

        # build predictor

        predictor = hippynn.Predictor([species,coords],[henergy.main_output, force],model_device='cuda:0',return_device='cuda:0')

        print("Configuration:",config)
        # warm up on first batch?
            
        for batch_size in tqdm(batch_size_list,desc='batch_sizes'):
            timing_list = []
            warm=False
            for rep in tqdm(range(n_reps_per_config+1),desc='reps'):
                torch.cuda.synchronize()
                time_start = time.time()
                # get time for predictor
                predictor.apply_to_database(db,batch_size=batch_size)
                torch.cuda.synchronize()
                time_end = time.time()
                this_time = time_end - time_start
                if not warm:
                    warm=True
                    continue
                timing_list.append(this_time)
            
            key = list(sorted(config.items(), key = lambda kv: kv[0])) # sort by keys in dictionary
            key = (*key, ("batch_size",batch_size))
            all_times[key] = timing_list
           
    device_name = torch.cuda.get_device_name()
    info = dict(metrics=all_times,device_name=device_name,n_configs=len(db.splits["test"]['coordinates']))

    print(info)

    torch.save(info,"speed_eval.pt")
