from training_script import load_db, make_model
import hippynn
hippynn.settings.WARN_LOW_DISTANCES=False
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable
import os
import time

from hippynn.graphs.gops import search_by_name

import torch



if __name__ == "__main__":

    n_reps_per_config = 5 # suggest 5 for production
    n_warmup_reps = 2

    batch_size_list = [int(os.environ.get("BATCH_SIZE", "2048"))]
    
    config_list=[
        dict(tensor_model="HOP", tensor_order=ell, tensor_factors=en) for ell in [3,2,1] for en in [4,3,2] if not (ell==1 and en > 2)
        ] + \
        [dict(tensor_model="NONE", tensor_order=0, tensor_factors=0)]

    if os.environ.get("HOP_CONFIGS"):
        config_list = [
            dict(tensor_model="HOP", tensor_order=int(spec.split(":")[0]), tensor_factors=int(spec.split(":")[1]))
            for spec in os.environ["HOP_CONFIGS"].split()
        ]
    elif os.environ.get("INCLUDE_L4_CONFIGS", "False").lower() in ("true", "1", "yes"):
        config_list += [
            dict(tensor_model="HOP", tensor_order=4, tensor_factors=3),
            dict(tensor_model="HOP", tensor_order=4, tensor_factors=4),
        ]

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
    anidata_location = "/vast/home/logan_bolton/Github/hippynn-optimizations-expanded/datasets/ani1x-release.h5"
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

        predictor_outputs = [henergy.main_output, force]
        if os.environ.get("INCLUDE_FORCES", "True").lower() in ("false", "0", "no"):
            predictor_outputs = [henergy.main_output]
        predictor = hippynn.Predictor([species,coords],predictor_outputs,model_device='cuda:0',return_device='cuda:0')

        print("Configuration:",config)
        # warm up on first batch?
            
        for batch_size in tqdm(batch_size_list,desc='batch_sizes'):
            timing_list = []
            for rep in tqdm(range(n_reps_per_config+n_warmup_reps),desc='reps'):
                torch.cuda.synchronize()
                time_start = time.time()
                # get time for predictor
                predictor.apply_to_database(db,batch_size=batch_size)
                torch.cuda.synchronize()
                time_end = time.time()
                this_time = time_end - time_start
                if rep < n_warmup_reps:
                    continue
                timing_list.append(this_time)
            
            key = list(sorted(config.items(), key = lambda kv: kv[0])) # sort by keys in dictionary
            key = (*key, ("batch_size",batch_size))
            all_times[key] = timing_list
           
    device_name = torch.cuda.get_device_name()
    info = dict(metrics=all_times,device_name=device_name,n_configs=len(db.splits["test"]['coordinates']))

    print(info)

    default_output = os.path.join(os.path.dirname(__file__), "..", "results", "nick", "speed_eval.pt")
    torch.save(info, os.environ.get("SPEED_EVAL_OUTPUT", default_output))
