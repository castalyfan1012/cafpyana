"""
Helper functions for working with the HDF5 df files.  

Each dataset (identified by a *key*) is stored as a set of smaller dfs (one per split),
so that very large samples can be handled in chunks.  
"""

import pandas as pd

def get_n_split(file):
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)

def load_dfs(file, keys2load, n_max_concat=100):
    out_df_dict = {}

    this_n_keys = get_n_split(file)
    n_concat = min(n_max_concat, this_n_keys)

    for key in keys2load:
        dfs = []

        for i in range(n_concat):
            try:
                this_df = pd.read_hdf(file, key=f"{key}_{i}")
                dfs.append(this_df)
            except KeyError:
                # stop if a split is missing
                break

        if len(dfs) == 0:
            print(f"[WARN] Missing or empty key: {key}")
            continue

        out_df_dict[key] = pd.concat(dfs, ignore_index=False)

    return out_df_dict
