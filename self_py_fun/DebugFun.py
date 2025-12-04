import numpy as np
import matplotlib.pyplot as plt


def produce_trun_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signals_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signals_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signals_cov, (E_val, length_per_electrode, length_per_electrode)

    note:
    -----
        descriptive mean and sample covariance statistics from real data
    """
    sample_size_len, feature_len = input_signal.shape

    input_signal = np.reshape(input_signal, [sample_size_len, E_val, int(feature_len/E_val)])
    input_signal_tar = input_signal[input_type == 1, ...]
    input_signal_ntar = input_signal[input_type == 0, ...]

    signal_tar_mean = np.mean(input_signal_tar, axis=0)
    signal_ntar_mean = np.mean(input_signal_ntar, axis=0)

    # Examine sample covariance matrix
    signal_tar_cov = np.stack([np.cov(input_signal_tar[:, e_iter, :], rowvar=False)
                                   for e_iter in range(E_val)], axis=0)
    signal_ntar_cov = np.stack([np.cov(input_signal_ntar[:, e_iter, :], rowvar=False)
                                    for e_iter in range(E_val)], axis=0)
    signal_all_cov = np.stack([np.cov(input_signal[:, e_iter, :], rowvar=False)
                                 for e_iter in range(E_val)], axis=0)
    return [signal_tar_mean, signal_ntar_mean,
            signal_tar_cov, signal_ntar_cov, signal_all_cov]


def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    r"""
    :param eeg_tar_mean:
    :param eeg_ntar_mean:
    :param subject_name:
    :param time_index:
    :param E_val:
    :param electrode_name_ls:
    :param y_limit: optional parameter, a list or an array of two numbers
    :param fig_size: optional parameter, a tuple of two numbers
    :return:
    """
    fig1, axes = plt.subplots(4, 4, figsize=fig_size)
    for e_iter in range(E_val):
        axes[int(e_iter / 4), e_iter % 4].plot(time_index, eeg_tar_mean[e_iter, :], color='red', label='Target')
        axes[int(e_iter / 4), e_iter % 4].plot(time_index, eeg_ntar_mean[e_iter, :], color='blue',
                                               label='Non-target')
        axes[int(e_iter / 4), e_iter % 4].legend(loc='upper right')
        axes[int(e_iter / 4), e_iter % 4].set_ylim(y_limit)
        axes[int(e_iter / 4), e_iter % 4].set_title(electrode_name_ls[e_iter])
    fig1.suptitle(subject_name)
    fig1.tight_layout()
    # plt.savefig("./{}/Mean.png".format(subject_name))
    plt.show()