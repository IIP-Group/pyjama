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
from tensorflow.python.keras.losses import MeanAbsoluteError, MeanSquaredError

from jammer.simulation_model import *
from jammer.utils import *
import jammer.simulation_model as sim

sim.EBN0_DB_MIN = -10
sim.EBN0_DB_MAX = 15
sim.NUM_SNR_POINTS = 26
sim.BATCH_SIZE = 512
sim.MAX_MC_ITER = 1500
sim.ebno_dbs = np.linspace(sim.EBN0_DB_MIN, sim.EBN0_DB_MAX, sim.NUM_SNR_POINTS)

if parameter_num == 0:
    sim.MAX_MC_ITER = 3000
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    decoder_parameters={}
    model_parameters["jammer_parameters"] = jammer_parameters
    model_parameters["decoder_parameters"] = decoder_parameters

    model_parameters["num_silent_pilot_symbols"] = 4
    jammer_parameters["num_tx"] = 2
    jammer_parameters["num_tx_ant"] = 2
    model_parameters["coderate"] = 0.5
    decoder_parameters["cn_type"] = "minsum"
    decoder_parameters["num_iter"] = 20

    model = Model(**model_parameters)
    model._decoder.llr_max = 1000
    simulate_model(model, f"No jammer", add_bler=True)

    model_parameters["jammer_present"] = True
    model = Model(**model_parameters)
    model._decoder.llr_max = 1000
    simulate_model(model, "Unmitigated jammer", add_bler=True)

    d = 4
    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = d
    model = Model(**model_parameters)
    simulate_model(model, f"Jammer, POS, {d}D Nulling", add_bler=True)

    ber_plots.title = "Different Decoder Parameters: 4UE, 0.5 coderate, 0dB Jammer"
    # ber_plots(show_ber=False)
    with open("bers/paper/frequency/mitigation_bler.pickle", "wb") as f:
        pickle.dump(ber_plots, f)
    sim.MAX_MC_ITER = 1500

elif parameter_num == 1:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["num_silent_pilot_symbols"] = 4
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = db_to_linear(10.)

    model = Model(**model_parameters)
    simulate_model(model, "Jammer, unmigitated, 0km/h")

    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = 1
    kmhs = [0, 20, 30, 40, 50, 60, 80, 100, 120]
    for kmh in kmhs:
        print("UE velocity: ", kmh)
        meter_per_second = kmh / 3.6
        model_parameters["min_ut_velocity"] = meter_per_second
        model_parameters["max_ut_velocity"] = meter_per_second
        # model_parameters["min_jammer_velocity"] = meter_per_second
        # model_parameters["max_jammer_velocity"] = meter_per_second
        model = Model(**model_parameters)
        simulate_model(model, f"Jammer, POS, {kmh} km/h")

    ber_plots.title = "UE velocity: Est. CSI, 1 UE, 1x1 Jammer (10dB)"
    # ber_plots()
    with open("bers/paper/frequency/ut_velocity_mitigation.pickle", "wb") as f:
        pickle.dump(ber_plots, f)

elif parameter_num == 2:
    ber_plots.reset()
    model_parameters = {}
    jammer_parameters = {}
    model_parameters["jammer_parameters"] = jammer_parameters

    model_parameters["num_silent_pilot_symbols"] = 4
    model_parameters["jammer_present"] = True
    model_parameters["jammer_power"] = db_to_linear(10.)

    model = Model(**model_parameters)
    simulate_model(model, "Jammer, unmigitated, 0km/h")

    model_parameters["jammer_mitigation"] = "pos"
    model_parameters["jammer_mitigation_dimensionality"] = 1
    kmhs = [0, 20, 30, 40, 50, 60, 80, 100, 120]
    for kmh in kmhs:
        print("Jammer velocity: ", kmh)
        meter_per_second = kmh / 3.6
        # model_parameters["min_ut_velocity"] = meter_per_second
        # model_parameters["max_ut_velocity"] = meter_per_second
        model_parameters["min_jammer_velocity"] = meter_per_second
        model_parameters["max_jammer_velocity"] = meter_per_second
        model = Model(**model_parameters)
        simulate_model(model, f"Jammer, POS, {kmh} km/h")

    ber_plots.title = "Jammer velocity: Est. CSI, 1 UE, 1x1 Jammer (10dB)"
    # ber_plots()
    with open("bers/paper/frequency/jammer_velocity_mitigation.pickle", "wb") as f:
        pickle.dump(ber_plots, f)
