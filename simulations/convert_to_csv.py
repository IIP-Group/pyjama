import pandas as pd
import numpy as np
import pickle
from sionna.utils.plotting import PlotBER
from itertools import compress
from datetime import datetime

def plotber_pickle_to_csv(pickle_path, save_path="./csv/", save_ber=True, save_bler=False):
    with open(pickle_path, 'rb') as f:
        plotber = pickle.load(f)
    if type(plotber) is not PlotBER:
        raise ValueError("The pickle file does not contain a PlotBER object")
    sim_title = pickle_path.split("/")[-1].rsplit(".", 1)[0]
    data = plotber_to_dict(plotber, save_ber, save_bler)
    save_data(sim_title, data, path=save_path)


def plotber_to_dict(plotber, save_ber=True, save_bler=False):
    snrs = plotber._snrs
    bers = plotber._bers
    legends = plotber._legends
    is_bler = plotber._is_bler
    
    if len(is_bler)>0: # ignore if object is empty
        if save_ber is False:
            snrs = list(compress(snrs, is_bler))
            bers = list(compress(bers, is_bler))
            legends = list(compress(legends, is_bler))
            is_bler = list(compress(is_bler, is_bler))

        if save_bler is False:
            snrs = list(compress(snrs, np.invert(is_bler)))
            bers = list(compress(bers, np.invert(is_bler)))
            legends = list(compress(legends, np.invert(is_bler)))
            is_bler = list(compress(is_bler, np.invert(is_bler)))

    result = {}
    # make sure all SNRs are the same, otherwise warn the user, and export all SNRs
    if all(np.allclose(x, snrs[0]) for x in snrs):
        result["snr_db"] = snrs[0]
    else:
        print("Not all SNRs are the same, exporting all SNRs")
        for i, snr in enumerate(snrs):
            result["snr_db " + legends[i]] = snr

    for i, ber in enumerate(bers):
        result[legends[i]] = ber
        
    return result
            

def save_data(sim_title, plot_data, sim_params=None, path="./csv/"):
    try:
        filename = (datetime.now().strftime("%Y-%m-%d %H-%M ")
                    + sim_title.replace("&", "")
                               .replace(".", "")
                               .replace(" ", "")
                               .replace("[","")
                               .replace("]","")
                               .replace(",","_"))
        file = open(path + filename + ".csv", "w")
        df = pd.DataFrame.from_dict(plot_data)
        df.to_csv(file, lineterminator='\n')
        file.close()

        with open(path + filename + ".pickle", 'wb') as f:
            pickle.dump(plot_data, f)

        if sim_params is not None:
            file = open(path + filename + '_params.csv', "w")
            df = pd.DataFrame.from_dict(sim_params)
            df.to_csv(file, lineterminator='\n')
            file.close()

    except Exception as e:
        print(e)


plotber_pickle_to_csv('bers/report/frequency/perf_vs_est_csi.pickle')