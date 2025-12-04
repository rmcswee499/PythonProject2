# import relevant packages
import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf # This will be used to create a PDF to store multiple plots in the same file

# Calculate the time-specific sample means and covariances for each individual electrode: 
def produce_trun_mean_cov(input_signal, input_type, E_val):
    
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)
        E_val: integer, (number of electrodes)

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signal_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_all_cov, (E_val, length_per_electrode, length_per_electrode)

    note:
    -----
        descriptive mean and sample covariance statistics from real data
        In this case, E_val=16, length_per_electrode=25. 
        But you should pass them as arguments or calculate them inside the function.
    """
    # Separate target and non-target signals
    target_signals_arr = input_signal[input_type == 1, :]
    nontarget_signals_arr = input_signal[input_type == -1, :]
    
    # Calculate number of samples per electrode
    length_per_electrode = int(input_signal.shape[1] / E_val)
    
    # Initialize output arrays
    signal_tar_mean = np.zeros((E_val, length_per_electrode))
    signal_ntar_mean = np.zeros((E_val, length_per_electrode))
    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    
    # Split by electrode (into E_val segments)
    target_by_electrode = np.array_split(target_signals_arr, E_val, axis=1)
    nontarget_by_electrode = np.array_split(nontarget_signals_arr, E_val, axis=1)

    # Loop over electrodes
    for e in range(E_val):
        # Compute mean for each electrode
        signal_tar_mean[e, :] = np.mean(target_by_electrode[e], axis=0)
        signal_ntar_mean[e, :] = np.mean(nontarget_by_electrode[e], axis=0)
        
        # Compute covariance for each electrode
        signal_tar_cov[e, :, :] = np.cov(target_by_electrode[e], rowvar=False)
        signal_ntar_cov[e, :, :] = np.cov(nontarget_by_electrode[e], rowvar=False)
        signal_all_cov[e, :, :] = np.cov(
            np.vstack((target_by_electrode[e], nontarget_by_electrode[e])),
            rowvar=False
        )      
    return(signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov)

# Plot the means for target and non-target signals for each individual electrode:

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
    fig, list_subfig = plt.subplots(nrows = 4, ncols = 4, figsize = fig_size)
    axes = list_subfig.flatten()
    
    for i in range(E_val):
        axes[i].plot(time_index, eeg_tar_mean[i,:], color = "red", label = "Target")
        axes[i].plot(time_index, eeg_ntar_mean[i,:], color = "blue", label = "Non-Target")
        axes[i].legend(loc='upper right')
        axes[i].set_title(electrode_name_ls[i])
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Amplitude (muV)")
    plt.suptitle("Subject: {}".format(subject_name))
    plt.tight_layout()  

    plt.savefig("/{}/Mean.png".format(subject_name))
    plt.show()  

# Plot the covariances for each electrode over time for a given patient (need to specify 'Target', 'Non-Target', or 'All'):

def plot_trunc_cov(
    eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14,12)
): 
    """
    Plots covariance matrices for each electrode in a 4x4 grid.

    Parameters:
        eeg_cov: np.ndarray, shape (E_val, n_timepoints, n_timepoints)
        cov_type: str, e.g. "Target" or "Non-Target"
        time_index: np.ndarray, time points in ms
        subject_name: str
        E_val: int, number of electrodes (16)
        electrode_name_ls: list of electrode names, length E_val
        fig_size: tuple, figure size
    """

    x_cov,y_cov = np.meshgrid(time_index, time_index) 
    fig_cov, cov_list_subfig = plt.subplots(nrows = 4, ncols = 4, figsize = fig_size) 
    axes = cov_list_subfig.flatten() 

    for i in range(E_val): 
        cs = axes[i].contourf(x_cov, y_cov, eeg_cov[i], cmap='ocean') 
        axes[i].set_ylim(max(time_index), min(time_index)) 
        axes[i].set_xlabel("Time (ms)") 
        axes[i].set_ylabel("Time (ms)") 
        axes[i].set_title(electrode_name_ls[i]) 
        plt.colorbar(cs) 
        plt.suptitle("{} Covariance for Subject: {}".format(cov_type,subject_name)) 
    
    plt.tight_layout() 

    plt.savefig("/{}/Covariance_{}.png".format(subject_name,cov_type)) 
    plt.show()  