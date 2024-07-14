import sys
gpu_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
parameter_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
# tf.config.run_functions_eagerly(True)
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy

from pyjama.simulation_model import *
from pyjama.utils import *
import pyjama.simulation_model as sim
from simulations.experimental_losses import *

sim.BATCH_SIZE = 128
sim.ebno_dbs = np.linspace(-5., 15., 21)
sim.MAX_MC_ITER = 500

if parameter_num == 0:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["num_ut"] = 1
    model_parameters["perfect_csi"] = True
    model = Model(**model_parameters)
    simulate_model(model, f"Perfect CSI")
    model_parameters["perfect_csi"] = False
    model = Model(**model_parameters)
    simulate_model(model, f"Estimated CSI")
    ber_plots.title = "1 UE: Perfect vs. Est. CSI"
    
    with open("bers/report/frequency/perf_vs_est_csi.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 1:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["perfect_csi"] = True
    model_parameters["jammer_present"] = True
    for jammer_power_db in [0, 3, 10]:
        for num_ut in [1, 8]:
            model_parameters["num_ut"] = num_ut
            model_parameters["jammer_power"] = db_to_linear(jammer_power_db)
            model = Model(**model_parameters)
            simulate_model(model, f"{num_ut} UEs, {jammer_power_db}dB jammer")

    ber_plots.title = "1 UE, Perf. CSI: Different Number of UEs"
    
    with open("bers/report/frequency/num_ues.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 2:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["perfect_csi"] = False
    model_parameters["perfect_jammer_csi"] = False
    model_parameters["jammer_present"] = True
    model_parameters["num_silent_pilot_symbols"] = 6

    model_parameters["jammer_mitigation"] = "ian"
    model = Model(**model_parameters)
    simulate_model(model, f"IAN, est. CSI")
    model_parameters["perfect_jammer_csi"] = True
    model = Model(**model_parameters)
    simulate_model(model, f"IAN, perf. jammer CSI")
    model_parameters["perfect_jammer_csi"] = False

    model_parameters["jammer_mitigation"] = "pos"
    model = Model(**model_parameters)
    simulate_model(model, f"POS")
    model_parameters["perfect_jammer_csi"] = True
    model = Model(**model_parameters)
    simulate_model(model, f"POS, perf. jammer CSI")

    ber_plots.title = "4 UE: IAN vs. POS"
    
    with open("bers/report/frequency/ian_vs_pos.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 3:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["perfect_csi"] = False
    model_parameters["perfect_jammer_csi"] = False
    model_parameters["jammer_present"] = True
    model_parameters["num_silent_pilot_symbols"] = 6
    model_parameters["jammer_mitigation"] = "pos"
    jammer_parameters["num_tx"] = 2
    jammer_parameters["num_tx_ant"] = 2

    for d in [1, 2, 3, 4, 5]:
        model_parameters["jammer_mitigation_dimensionality"] = d
        model = Model(**model_parameters)
        simulate_model(model, f"{d}D Nulling")

    ber_plots.title = "2x2 Jammer: Dimensionality of Mitigation (est. CSI)"
    
    with open("bers/report/frequency/mitigation_dim.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 4:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["perfect_csi"] = False
    model_parameters["num_ut"] = 1
    model_parameters["jammer_present"] = True
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = 1

    model_parameters["perfect_jammer_csi"] = True
    model = Model(**model_parameters)
    simulate_model(model, f"Perfect Jammer CSI")

    model_parameters["perfect_jammer_csi"] = False
    for i in range(1, 8):
        model_parameters["num_silent_pilot_symbols"] = i
        model = Model(**model_parameters)
        simulate_model(model, f"{i} Silent Symbols")

    ber_plots.title = "1 UE, Est. CSI: Different Number of Silent Symbols"
    
    with open("bers/report/frequency/num_silent.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 5:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["num_ut"] = 1
    model_parameters["perfect_csi"] = False
    model_parameters["perfect_jammer_csi"] = False
    model_parameters["num_silent_pilot_symbols"] = 6
    jammer_parameters["num_tx_ant"] = 1
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = db_to_linear(10.)

    model = Model(**model_parameters)
    simulate_model(model, "Jammer, unmigitated, 0km/h")

    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = 2
    kmhs = [0, 20, 120]
    for kmh in kmhs:
        meter_per_second = kmh / 3.6
        model_parameters["min_ut_velocity"] = meter_per_second
        model_parameters["max_ut_velocity"] = meter_per_second
        model_parameters["min_jammer_velocity"] = meter_per_second
        model_parameters["max_jammer_velocity"] = meter_per_second
        model = Model(**model_parameters)
        simulate_model(model, f"Jammer, POS, {kmh} km/h")

    ber_plots.title = "Jammers with velocity: Est. CSI, 1 UE, 1x1 Jammer (10dB)"
    
    with open("bers/report/frequency/velocity_mitigation.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 6:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters
    model_parameters["cyclic_prefix_length"] = 0

    model_parameters["num_ut"] = 1
    model_parameters["perfect_csi"] = False
    model_parameters["perfect_jammer_csi"] = False
    model_parameters["num_silent_pilot_symbols"] = 6
    jammer_parameters["num_tx_ant"] = 1
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = db_to_linear(10.)
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = 1
    model = Model(**model_parameters)
    simulate_model(model, "No Mobility")
    velocity_kmh = 120
    velocity_mps = velocity_kmh / 3.6
    model_parameters["min_ut_velocity"] = velocity_mps
    model_parameters["max_ut_velocity"] = velocity_mps
    model_parameters["min_jammer_velocity"] = velocity_mps
    model_parameters["max_jammer_velocity"] = velocity_mps

    model = Model(**model_parameters)
    simulate_model(model, f"Standard")
    model = Model(carrier_frequency=4.5e8, **model_parameters)
    simulate_model(model, f"450MHz Carrier Frequency")
    model = Model(subcarrier_spacing=240e3, fft_size=16, **model_parameters)
    simulate_model(model, f"240kHz Subcarrier Spacing (but same BW)")
    model = Model(fft_size=1024, **model_parameters)
    simulate_model(model, f"8x BW")

    ber_plots.title = "Mobility: Analysis of different factors"
    
    with open("bers/report/frequency/velocity_factors.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

if parameter_num == 7:
    # TODO change iterations
    sim.MAX_MC_ITER = 2000
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    decoder_parameters={}
    model_parameters["jammer_parameters"] = jammer_parameters
    model_parameters["decoder_parameters"] = decoder_parameters

    model_parameters["perfect_csi"] = False
    model_parameters["num_ut"] = 4
    model_parameters["coderate"] = 0.5
    decoder_parameters["cn_type"] = "boxplus-phi"
    decoder_parameters["num_iter"] = 20

    model = Model(**model_parameters)
    simulate_model(model, "Boxplus-Phi, LLR cutoff, no Jammer")

    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = db_to_linear(0.)

    model = Model(**model_parameters)
    simulate_model(model, "Boxplus-Phi, LLR cutoff, Uniform Jammer", add_bler=True)

    model = Model(**model_parameters)
    model._decoder.llr_max = 1000
    simulate_model(model, "Boxplus-Phi, no LLR cutoff, Uniform Jammer", add_bler=True)

    decoder_parameters["cn_type"] = "minsum"
    model = Model(**model_parameters)
    simulate_model(model, "Minsum, LLR cutoff, Uniform Jammer", add_bler=True)

    model = Model(**model_parameters)
    model._decoder.llr_max = 1000
    simulate_model(model, "Minsum, no LLR cutoff, Uniform Jammer", add_bler=True)

    ber_plots.title = "Different Decoder Parameters: 4UE, 0.5 coderate, 0dB Jammer"
    
    with open("bers/report/frequency/bler_params.pickle", "wb") as f:
        pickle.dump(ber_plots, f)