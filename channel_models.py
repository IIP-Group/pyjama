#%%
import tensorflow as tf
import sionna
from sionna.channel import ChannelModel
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology
import matplotlib.pyplot as plt
import numpy as np

class MultiTapRayleighBlockFading(ChannelModel):
    # pylint: disable=line-too-long
    r"""RayleighBlockFading(num_rx, num_rx_ant, num_tx, num_tx_ant, dtype=tf.complex64)

    Generate channel impulse responses corresponding to a Rayleigh block
    fading channel model.

    The channel impulse responses generated are formed of M paths with
    :math:`m \div sampling_frequency, 0 \leq m \leq M-1` delays
    and a normally distributed fading coefficient.
    All time steps of a batch example share the same channel coefficient
    (block fading).

    This class can be used in conjunction with the classes that simulate the
    channel response in time or frequency domain, i.e.,
    :class:`~sionna.channel.OFDMChannel`,
    :class:`~sionna.channel.TimeChannel`,
    :class:`~sionna.channel.GenerateOFDMChannel`,
    :class:`~sionna.channel.ApplyOFDMChannel`,
    :class:`~sionna.channel.GenerateTimeChannel`,
    :class:`~sionna.channel.ApplyTimeChannel`.

    Parameters
    ----------

    num_rx : int
        Number of receivers (:math:`N_R`)

    num_rx_ant : int
        Number of antennas per receiver (:math:`N_{RA}`)

    num_tx : int
        Number of transmitters (:math:`N_T`)

    num_tx_ant : int
        Number of antennas per transmitter (:math:`N_{TA}`)

    num_paths: int
        Number of paths (:math:`M`)

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----
    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps
        
    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(   self,
                    num_rx,
                    num_rx_ant,
                    num_tx,
                    num_tx_ant,
                    num_paths,
                    dtype=tf.complex64):

        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype

        # We don't set these attributes as private so that the user can update
        # them
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        self.num_paths = num_paths

    def __call__(self,  batch_size, num_time_steps, sampling_frequency):

        # Delays
        delays = tf.range(0, self.num_paths, dtype=self._dtype.real_dtype) / sampling_frequency
        delays = tf.tile(delays[tf.newaxis, tf.newaxis, tf.newaxis, :], [batch_size, self.num_rx, self.num_tx, 1])

        # Fading coefficients
        std = tf.cast(tf.sqrt(0.5), dtype=self._dtype.real_dtype)
        h_real = tf.random.normal(shape=[   batch_size,
                                            self.num_rx,
                                            self.num_rx_ant,
                                            self.num_tx,
                                            self.num_tx_ant,
                                            self.num_paths,
                                            1], # Same response over the block
                                            stddev=std,
                                            dtype = self._dtype.real_dtype)
        h_img = tf.random.normal(shape=[    batch_size,
                                            self.num_rx,
                                            self.num_rx_ant,
                                            self.num_tx,
                                            self.num_tx_ant,
                                            self.num_paths,
                                            1], # Same response over the block
                                            stddev=std,
                                            dtype = self._dtype.real_dtype)
        h = tf.complex(h_real, h_img)
        # Tile the response over the block
        h = tf.tile(h, [1, 1, 1, 1, 1, 1, num_time_steps])
        return h, delays


def visualize_cir(a, tau):
    plt.scatter(tau[0,0,0,:]/1e-9, np.abs(a)[0,0,0,0,0,:,0])
    plt.xlabel(r"$\tau$ [ns]")
    plt.ylabel(r"$|a|$")

    # plt.figure()
    # for i in range(a.shape[-2]):
    #     plt.plot(np.abs(a)[0,0,0,0,0,i,:], label=f"Path {i}")
    # plt.legend()

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
        "enable_pathloss": False,
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

def visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, topology=None, indoor_probability=0.8, los=None):
    """either provide topology or indoor_probability"""
    channel = setup_3gpp_channel(scenario=scenario, carrier_frequency=carrier_frequency)
    if topology is None:
        topology = new_topology(scenario=scenario, indoor_probability=indoor_probability)
    channel.set_topology(*topology, los=los)
    # TODO for now, neither num_samples nor sampling frequency matter(?)
    cir = channel(1000, carrier_frequency)
    cir = filter_cir(*cir)
    visualize_cir(*cir)

topology = new_topology(scenario="umi", indoor_probability=0.0)
for i in range(100):
    print(i)
    visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, topology=topology, los=True)
plt.show()
# plt.figure()
# visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, topology=topology)
# plt.figure()
# visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, topology=topology)
# plt.figure()
# visualize_3gpp_channel(scenario="umi", carrier_frequency=3.5e9, topology=topology)

# %%
