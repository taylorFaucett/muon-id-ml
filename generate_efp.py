import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob
from os import path, getcwd
from tqdm import tqdm, trange
from itertools import product


def efp(data, graph, kappa, beta, normed):
    EFP_graph = ef.EFP(graph, measure="hadr", kappa=kappa, beta=beta, normed=normed)
    return EFP_graph.batch_compute(data)


def generate_EFP():
    hdf_file = path.join(data_path, "processed", "prep_data.h5")
    X = pd.read_hdf(hdf_file, "features").features.to_numpy()
    y = pd.read_hdf(hdf_file, "targets").targets.values
    
    # Choose kappa, beta values
    kappas = [-1, 0, 0.25, 0.5, 1, 2]
    betas = [0.25, 0.5, 1, 2, 3, 4]

    # Grab graphs
    prime_d7 = ef.EFPSet("d<=7", "p==1")
    chrom_4 = ef.EFPSet("d<=8", "p==1", "c==4")
    efpsets = [prime_d7, chrom_4]
    for efpset in efpsets:
        graphs = efpset.graphs()
        t = tqdm(graphs)
        for efp_ix, graph in enumerate(t):
            for kappa in kappas:
                for beta in betas:
                    n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
                    file_name = f"data/efp/efp_{n}_{d}_{k}_k_{kappa}_b_{beta}.feather"
                    if not path.isfile(file_name):
                        graph = graphs[efp_ix]
                        t.set_description(
                            f"Procesing: EFP[{n},{d},{k}](k={kappa},b={beta})"
                        )
                        efp_val = efp(
                            data=X,
                            graph=graph,
                            kappa=kappa,
                            beta=beta,
                            normed=False,
                        )
                        efp_df = pd.DataFrame(
                            {f"features": efp_val, f"targets": y}
                        )
                        efp_df.to_feather(file_name)



if __name__ == "__main__":
    home = getcwd()
    data_path = path.join(home, "data")
    generate_EFP()
