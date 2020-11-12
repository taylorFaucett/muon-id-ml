import numpy as np
import pandas as pd
from os import path, getcwd
from tqdm import tqdm, trange


def generate_prep_data():
    # Load jet-images
    X = np.load(path.join(data_path, "raw", "images.npy"))
    y = np.load(path.join(data_path, "raw", "labels.npy"))
    df0 = []
    
    # This is based on edges at 0.4 but the mid-point of the cells being 0.8/32/2 away from the edge
    eta_ = np.linspace(-0.3875, 0.3875, X.shape[2])
    phi_ = np.linspace(
        -(15.5 * np.pi) / 126.0, (15.5 * np.pi) / 126.0, X.shape[2]
    )
    
    #     cell_edge = 0.3875
    #     eta_ = np.linspace(-cell_edge, cell_edge, X.shape[2]) 
    #     phi_ = np.linspace(-cell_edge, cell_edge, X.shape[2])

    eta_phi = np.vstack([(x, y) for x in eta_ for y in phi_])
    eta_ = eta_phi[:, 0]
    phi_ = eta_phi[:, 1]
    for ix in trange(len(X)):
        et = X[ix].flatten()
        dfi = pd.DataFrame({"et": et, "eta": eta_, "phi": phi_})
        evt_out = dfi[(dfi[["et"]] != 0).all(axis=1)].to_numpy()
        evt_out[:, 0] /= np.sum(evt_out[:, 0])
        df0.append(evt_out)
    X0 = pd.DataFrame({"features": df0})
    y0 = pd.DataFrame({"targets": y})
    hdf_out = path.join(data_path, "processed", "prep_data.h5")
    X0.to_hdf(hdf_out, "features", mode="a")
    y0.to_hdf(hdf_out, "targets", mode="a")


if __name__ == "__main__":
    """ output: feature and target files
    
    generate_prep_data converts a jet image into [ET, eta, phi] format to be 
    processed by the energyflow package.
    """
    home = getcwd()
    data_path = path.join(home, "data")
    generate_prep_data()
