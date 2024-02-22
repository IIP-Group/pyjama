import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology
from .visualization_utils import axis_add_custom_ticks
from matplotlib.ticker import FormatStrFormatter, FixedFormatter, FixedLocator, MultipleLocator


def visualize_cir(a, tau):
    plt.scatter(tau[0,0,0,:], np.square(np.abs(a))[0,0,0,0,0,:,0])
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(r"$|a|^2$")

def setup_3gpp_channel(scenario, carrier_frequency):
    # TODO refactor this for less code duplication (cmp. jammer_simulation.py:Model._generate_channel)
    channel_type_to_class = {
        "umi": UMi,
        "uma": UMa,
        "rma": RMa,
    }
    ut_array = AntennaArray(
                        num_rows=1,
                        num_cols=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="omni",
                        carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1,
                        num_cols=1,
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
    channel_parameters = {
        "carrier_frequency": carrier_frequency,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": "uplink",
        "enable_pathloss": True,
        "enable_shadow_fading": False,
    }
    if scenario in ["umi", "uma"]:
        channel_parameters["o2i_model"] = "low"
    channel = channel_type_to_class[scenario](**channel_parameters)
    return channel

def new_topology(scenario, indoor_probability=0.8):
    return gen_single_sector_topology(1,
                                      1,
                                      scenario,
                                      min_ut_velocity=0.0,
                                      max_ut_velocity=0.0,
                                      indoor_probability=indoor_probability)

def filter_cir(a, tau):
    # filter paths where a is not zero over all time samples
    energy_sum = tf.reduce_sum(tf.abs(a)**2, axis=[0,1,2,3,4,6])
    nonzero_paths = sionna.utils.flatten_last_dims(tf.where(energy_sum > 0.0))
    a = tf.gather(a, nonzero_paths, axis=-2)
    tau = tf.gather(tau, nonzero_paths, axis=-1)
    return a, tau

def visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, indoor_probability=0.8, los=None, resample_topology=False, num_cir_samples=2000, num_bins=200):
    """either provide topology or indoor_probability"""
    channel = setup_3gpp_channel(scenario=scenario, carrier_frequency=carrier_frequency)
    topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
    channel.set_topology(*topology, los=los)
    # 1.: scatter plot of 4 
    plt.suptitle(f"Scenario: {scenario}, indoor probability: {indoor_probability}, LoS: {los}")
    plt.gcf().set_size_inches(4.5, 18)
    plt.subplot(4,1,1)
    plt.title("4 CIRs")
    for i in range(4):
        # TODO for now, neither num_samples nor sampling frequency matter(?)
        cir = channel(1, carrier_frequency)
        cir = filter_cir(*cir)
        visualize_cir(*cir)
        if resample_topology:
            topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
            channel.set_topology(*topology, los=los)
    # 2.: empirical distribution of delays tau (number of occurrences)
    # 3. empirical distribution of amplitudes |a| (mean power)
    delays = np.array([])
    delay_power = np.empty((2, 0))
    for i in range(num_cir_samples):
        if i % 20 == 0:
            print(i)
        cir = channel(1, carrier_frequency)
        a, tau = filter_cir(*cir)
        # 2.
        delays = np.concatenate([delays, tau.numpy().flatten()])
        # 3.
        a_squeezed = tf.squeeze(a)
        # just treat antennas as multiple datapoints
        tau_squeezed = tf.broadcast_to(tf.squeeze(tau), a_squeezed.shape)
        a_squeezed = sionna.utils.flatten_last_dims(a_squeezed)
        tau_squeezed = sionna.utils.flatten_last_dims(tau_squeezed)
        # x is tau, y is |a|^2
        power_points = tf.stack([tau_squeezed, tf.abs(a_squeezed)**2], axis=0)
        delay_power = np.concatenate([delay_power, power_points.numpy()], axis=1)
        if resample_topology:
            topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
            channel.set_topology(*topology, los=los)
    plt.subplot(4,1,2)
    plt.title("Delay distribution: Histogram and CDF")
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel("Number of impulses")
    plt.hist(delays, bins=num_bins)
    plt.twinx()
    plt.ylabel("CDF")
    plt.ecdf(delays, color="C1")
    plt.subplot(4,1,3)
    plt.title(f"Scatter plot of {num_cir_samples} CIRs")
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(r"$|a|^2$")
    plt.scatter(delay_power[0,:], delay_power[1,:], s=0.1)
    # average power in delay bins
    max_delay = np.max(delay_power[0,:])
    delay_bins = np.linspace(0, max_delay, num_bins+1)
    power_bins = np.empty(num_bins)
    for i in range(num_bins):
        mask = np.logical_and(delay_power[0,:] >= delay_bins[i], delay_power[0,:] < delay_bins[i+1])
        power_bins[i] = np.mean(delay_power[1,mask])
    power_bins = np.nan_to_num(power_bins)
    plt.subplot(4,1,4)
    plt.title("Average power")
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(r"$|a|^2$")
    plt.plot(delay_bins[:-1], power_bins)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()





# below 802.11n
# TODO: don't confuse sampling frequency with spacing
def presentation(scenario="umi", carrier_frequency=3.5e9, sampling_time=3.255e-8, indoor_probability=0.8, los=None, resample_topology=False, num_cir_samples=2000, num_bins=200):
    """either provide topology or indoor_probability"""
    channel = setup_3gpp_channel(scenario=scenario, carrier_frequency=carrier_frequency)
    topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
    channel.set_topology(*topology, los=los)
    # 2.: empirical distribution of delays tau (number of occurrences)
    # 3. empirical distribution of amplitudes |a| (mean power)
    delays = np.array([])
    delay_power = np.empty((2, 0))
    for i in range(num_cir_samples):
        if i % 20 == 0:
            print(i)
        cir = channel(1, 1/sampling_time)
        a, tau = filter_cir(*cir)
        # 2.
        delays = np.concatenate([delays, tau.numpy().flatten()])
        # 3.
        a_squeezed = tf.squeeze(a)
        # just treat antennas as multiple datapoints
        tau_squeezed = tf.broadcast_to(tf.squeeze(tau), a_squeezed.shape)
        a_squeezed = sionna.utils.flatten_last_dims(a_squeezed)
        tau_squeezed = sionna.utils.flatten_last_dims(tau_squeezed)
        # x is tau, y is |a|^2
        power_points = tf.stack([tau_squeezed, tf.abs(a_squeezed)**2], axis=0)
        delay_power = np.concatenate([delay_power, power_points.numpy()], axis=1)
        if resample_topology:
            topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
            channel.set_topology(*topology, los=los)
    plt.figure()
    ax = plt.gca()
    # ax.tick_params(axis='x', labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(0.5e-5))

    plt.title("Delay distribution: Power and impulse CDF")
    plt.ylabel("CDF")
    plt.ecdf(delays, color="C1")
    plt.twinx()
    #add ticks for samping frequency
    axis_add_custom_ticks(ax.xaxis, ticks={100 * sampling_time: '100T'})
    ax.get_xticklabels()[2].set_horizontalalignment('right')
    # average power in delay bins
    # max_delay = np.max(delay_power[0,:])
    max_delay = np.max(delay_power[0,:])
    delay_bins = np.linspace(0, max_delay, num_bins+1)
    power_bins = np.empty(num_bins)
    for i in range(num_bins):
        mask = np.logical_and(delay_power[0,:] >= delay_bins[i], delay_power[0,:] < delay_bins[i+1])
        power_bins[i] = np.mean(delay_power[1,mask])
    power_bins = np.nan_to_num(power_bins)
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(r"$|a|^2$")
    plt.plot(delay_bins[:-1], power_bins)

    # name='umi_plot_5G'
    # plt.savefig(f"{name}.png")
    plt.show()

def visualize_channel_filter_taps(scenario="umi",
                                  carrier_frequency=3.5e9,
                                  bandwidth=2.e7,
                                  indoor_probability=0.8,
                                  los=None,
                                  resample_topology=True,
                                  num_cir_samples=2000):

    l_min, l_max = sionna.channel.time_lag_discrete_time_channel(bandwidth)
    channel = setup_3gpp_channel(scenario=scenario, carrier_frequency=carrier_frequency)
    topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
    channel.set_topology(*topology, los=los)

    hms = []
    for i in range(num_cir_samples):
        if i % 20 == 0:
            print(f"Sample: {i}", end='\r')
        cir = channel(1, bandwidth)
        hm = sionna.channel.cir_to_time_channel(bandwidth, *cir, l_min, l_max, normalize=True)
        hms.append(hm[0,0,0,0,0,0])
        if resample_topology:
            topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
            channel.set_topology(*topology, los=los)

    hm_avg = np.mean(np.square(np.abs(hms)), axis=0)
    plt.figure(figsize=(10, 7.5))
    bw_mhz = bandwidth / 1e6
    plt.title(f"{scenario} @ {bw_mhz}MHz, indoor: {indoor_probability}, los: {los}")
    # plt.title("UMi @ 20MHz: Avg. Energy of Discrete Time Channel Taps")
    plt.xlabel("$\ell$ (time lag)")
    plt.ylabel(r"$|\bar{h}_{0,\ell}|^2$")
    plt.stem(range(l_min, l_max+1), hm_avg)
    plt.semilogy()
    plt.show()

    

# presentation('umi', indoor_probability=0.8, los=False, resample_topology=True, num_cir_samples=2000, num_bins=200)
# presentation('umi', indoor_probability=0.8, los=False, resample_topology=True, num_cir_samples=100, num_bins=100)

# for scenario in ["umi", "uma", "rma"]:
#     visualize_3gpp_channel(scenario=scenario, carrier_frequency=3.5e9, indoor_probability=0.8, los=None,
#                            num_cir_samples=2000, num_bins=200, resample_topology=True)
# for los in [True, False]:
#     for indoor_probability in [0.0, 1.0]:
#         visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, indoor_probability=indoor_probability, los=los,
#                                num_cir_samples=2000, num_bins=200, resample_topology=True)

 # %%
