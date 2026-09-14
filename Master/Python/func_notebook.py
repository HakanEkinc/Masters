import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import uproot
import awkward
import re
from pathlib import Path
import math
import seaborn as sns



#Sample window for 64CH digitizer 
baseline_start_64 = 0 
baseline_end_64 = 530
peak_start_64 = 564
peak_end_64 = 670

#Folders where the data will be stored
parquet_folder = "/disk/gfs_atp/xlzd/alpine/analysis/parquets"
plot_folder = "/disk/gfs_atp/xlzd/alpine/analysis/plots"

####################################################################
#ROOT DATA 
####################################################################


#input the root file of the DAQ measurement
#output a dataframe with channels, baseline, sigmas, area of the peak
def analyse_root_data(filepath, new_path = None):
    with uproot.open(filepath) as file:
        tree = file['Events']
        
        # Changed to library="np". 
        # This returns a dictionary: {"ch01": array, "ch02": array, ...}
        # If your events are all the exact same length, these are ALREADY 2D matrices!
        data_dict = tree.arrays(filter_name="/^ch[0-9]+$/", library="np")
    
    analyzed_data = {}
    
    for channel_name, waveforms in data_dict.items():
        print(f"Processing {channel_name}...")
        
        # If your data comes as an array of objects (jagged), we just stack it
        if waveforms.dtype == 'O': 
            waveforms = np.vstack(waveforms)
            
        # 2. Define the sample windows (waveforms is already a 2D matrix)
        baseline_window = waveforms[:, baseline_start_64:baseline_end_64] 
        peak_window = waveforms[:, peak_start_64:peak_end_64]
        
        # 3. Compute baseline and standard deviation
        baselines = np.mean(baseline_window, axis=1)
        std_devs = np.std(baseline_window, axis=1)
        
        baseline_corrected_peak = peak_window - baselines[:, None]
        #check if signal is positive (flip if necessary)
        if np.mean(baseline_corrected_peak) < 0:
            print(f"  -> Negative polarity detected on {channel_name}. Auto-inverting.")
            baseline_corrected_peak *= -1
        
        # 4. Integrate the area
        
        integrals = np.sum(baseline_corrected_peak, axis=1)
        
        # 5. Save the results into our dictionary
        analyzed_data[channel_name] = pd.DataFrame({
            "channel": channel_name,
            "baseline": baselines,
            "std_dev": std_devs,
            "peak_integral": integrals
        })

    master_df = pd.concat(analyzed_data.values(), ignore_index=True)
    
    # 2. Check if a parquet_name was provided 
    if new_path is not None:


        original_name = Path(filepath).parent.name
        final_parquet_path = Path(new_path) / f"{original_name}.parquet"
        final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the combined DataFrame
        master_df.to_parquet(final_parquet_path, index=False)
        print(f"Data successfully saved  to {final_parquet_path}")
        
    return master_df, final_parquet_path






# --------------------------------------------------------- 
# 1. Physics-Linked Mathematical Model 
# --------------------------------------------------------- 

def fit_model_physics(x, *params): 
    """ 
    Physics-linked model for SiPM/PMT multi-photon spectra. 
    params[0]: A_bg       (Exponential background amplitude) 
    params[1]: lambda_bg  (Exponential background decay rate) 
    params[2]: mu_0       (Position of the 0-photon pedestal) 
    params[3]: sigma_0    (Width of the 0-photon pedestal) 
    params[4]: gain       (Distance between adjacent photon peaks) 
    params[5]: sigma_1    (Additional width contribution per photon) 
    params[6]: C          (Constant baseline offset)
    params[7:]: A_0, A_1, A_2... (Amplitudes for each photon peak) 
    """ 
    A_bg = params[0] 
    lambda_bg = params[1] 
    mu_0 = params[2] 
    sigma_0 = params[3] 
    gain = params[4] 
    sigma_1 = params[5] 
    C = params[6]
    amplitudes = params[7:] 
    
    # Calculate Background (Shifted Exponential + Constant Baseline)
    y = np.where(x >= mu_0, A_bg * np.exp(-lambda_bg * (x - mu_0)), 0.0) + C
    
    # Add Physics-Linked Gaussians 
    for n, A in enumerate(amplitudes): 
        mu_n = mu_0 + n * gain 
        sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2) 
        y += A * np.exp(-0.5 * ((x - mu_n) / sigma_n)**2) 
        
    return y


def fit_pandas_data(raw_data, channel_name="Data", bins=500, num_peaks_to_fit=18, 
                    confident_peak_limit=7, manual_mu0=None, manual_gain=None, ax=None,
                    max_reduced_chi2=None): 
    
    raw_data = np.array(raw_data.dropna(), dtype=float) 
    print(f"Processing {len(raw_data)} data points for {channel_name}...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 
    
    fit_successful = False
    popt, pcov = None, None
    peak_params = [] 
    
    reduced_chi_square = np.nan
    chi_square = np.nan
    ndf = 0

    if manual_mu0 is not None and manual_gain is not None: 
        print("Using manual physics overrides for Pedestal and Gain.") 
        mu_0_guess = manual_mu0 
        gain_guess = manual_gain 
    else: 
        print("Auto-detecting peaks...") 
        smoothed_counts = gaussian_filter1d(counts, sigma=1) 
        peaks, _ = find_peaks( 
            smoothed_counts,  
            prominence=np.max(smoothed_counts) * 0.01,  
            distance=4 
        ) 
        
        global_max_bin = np.argmax(counts) 
        if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
            peaks = np.append(peaks, global_max_bin) 
            peaks = np.sort(peaks) 

        if len(peaks) == 0: 
            print("Failed to find any peaks. Skipping fit and plotting raw data.")
        else:
            idx_0 = peaks[0] 
            mu_0_guess = bin_centers[idx_0] 
            
            if len(peaks) > 1: 
                gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
            else: 
                gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

    if (manual_mu0 is not None) or (len(peaks) > 0):
        sigma_0_guess = max(dx, 1e-9) 
        sigma_1_guess = sigma_0_guess * 0.5  

        p0 = [] 
        lower_bounds = [] 
        upper_bounds = [] 

        A_bg_guess = np.max(counts) * 0.1 
        lambda_bg_guess = 1.0 / max(np.abs(np.mean(raw_data)), 1.0) 
        p0.extend([A_bg_guess, lambda_bg_guess]) 
        lower_bounds.extend([0, 0]) 
        upper_bounds.extend([np.max(counts) * 0.5, 5000]) 

        p0.extend([mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
        lower_bounds.extend([bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
        upper_bounds.extend([bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

        # Add Baseline Constant Bounds (Index 6)
        C_guess = 1.0
        p0.append(C_guess)
        lower_bounds.append(0.0)
        upper_bounds.append(np.max(counts) * 0.1)

        for n in range(num_peaks_to_fit): 
            expected_mu = mu_0_guess + (n * gain_guess) 
            if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
                closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
                A_guess = counts[closest_bin_idx] 
            else: 
                A_guess = 1e-9  
                
            p0.append(A_guess) 
            lower_bounds.append(0.0) 
            upper_bounds.append(np.inf) 

        p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

        # Poisson error + 2% systematic error floor
        y_err = np.maximum(np.sqrt(counts), 1.0) + (0.02 * counts)

        try: 
            popt_temp, pcov_temp = curve_fit( 
                fit_model_physics,  
                bin_centers,  
                counts,  
                p0=p0, 
                sigma=y_err,          
                absolute_sigma=True,  
                bounds=(lower_bounds, upper_bounds), 
                maxfev=25000  
            ) 
            
            expected_counts = fit_model_physics(bin_centers, *popt_temp)
            
            # Use the exact same weighting used in the optimizer for chi-square
            variances = y_err ** 2
            
            chi_square = np.sum(((counts - expected_counts) ** 2) / variances)
            ndf = len(counts) - len(popt_temp)
            reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
            
            if max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                print(f"Fit rejected: Reduced \u03C7\u00B2 ({reduced_chi_square:.2f}) > threshold ({max_reduced_chi2}).")
            else:
                print("Physics Curve fitting converged successfully!") 
                fit_successful = True
                popt = popt_temp
                pcov = pcov_temp
                
                # Unpack the 7 physical parameters
                A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1, C = popt[0:7] 
                amplitudes = popt[7:] 
                
                for n in range(len(amplitudes)):
                    mu_n = mu_0 + n * gain
                    sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2)
                    peak_params.append([mu_n, sigma_n])
            
        except Exception as e: 
            print(f"Optimal parameters not found: {e}") 

    if ax is not None:
        ax.hist(raw_data, bins=bins, alpha=0.5, color='blue', label='Binned Data') 
        
        if fit_successful:
            x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
            ax.plot(x_fit, fit_model_physics(x_fit, *popt), color='red', lw=2, label='Total Fit') 
            
            print(f"\n--- Physical Fit Results for {channel_name} ---") 
            for n, (mu_n, sigma_n) in enumerate(peak_params[:4]):
                print(f"  {n}-Photon Peak: \u03BC = {mu_n:.6e}, \u03C3 = {sigma_n:.6e}")
            print("-" * 50) 
            print(f"\u03C7\u00B2 / NDF:           {chi_square:.2f} / {ndf} = {reduced_chi_square:.3f}")
            print("-" * 28) 
            
            tail_sum = np.zeros_like(x_fit) 
            for n, (mu_n, sigma_n) in enumerate(peak_params): 
                A = popt[7 + n]
                single_gauss = A * np.exp(-0.5 * ((x_fit - mu_n) / sigma_n)**2) 
                
                if n <= confident_peak_limit: 
                    if np.max(single_gauss) > (np.max(counts) * 0.005): 
                        ax.plot(x_fit, single_gauss, '--', label=f'{n}-Photon') 
                else: 
                    tail_sum += single_gauss 

            if np.max(tail_sum) > (np.max(counts) * 0.005): 
                ax.plot(x_fit, tail_sum, color='gray', linestyle='-.', lw=1.5, label='High-Photon Tail') 

            ax.set_title(f'{channel_name}, with \u03C7\u00B2/NDF = {reduced_chi_square:.3f}') 
        else:
            if not np.isnan(reduced_chi_square) and max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                ax.set_title(f'{channel_name} (REJECTED: \u03C7\u00B2/NDF > {max_reduced_chi2})') 
            else:
                ax.set_title(f'{channel_name} (FIT FAILED)') 

        ax.set_xlabel('ADC Counts') 
        ax.set_ylabel('Frequency') 
        ax.grid(alpha=0.3) 

    return popt, pcov, np.array(peak_params)




def process_all_channels_to_single_canvas(data_dict, output_pdf_path="all_channels_summary.pdf"):
    """
    data_dict: dict mapping channel_name to raw_pandas_data 
               e.g. {"Ch1": df['ch1'], "Ch2": df['ch2'], ...}
    """
    num_plots = len(data_dict)
    if num_plots == 0:
        return
        
    # Calculate grid dimensions (max 10 columns per row)
    cols = min(10, num_plots)
    rows = math.ceil(num_plots / 10)
    
    # Create the single large canvas. Adjust figsize per subplot as needed.
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 3.5 * rows))
    
    # Flatten axes array for easy 1D iteration if it's a grid
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Handle single plot case safely
        
    # Iterate through your data and plot each to its specific axis
    for idx, (channel_name, raw_data) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        # Call the updated function, passing the specific axis
        fit_pandas_data(
            raw_data=raw_data, 
            channel_name=channel_name, 
            ax=ax 
        )
        
    # Hide any unused axes if num_plots isn't a perfect multiple of 10
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
        
    # Finalize and save the single canvas
    plt.tight_layout()
    plt.savefig(output_pdf_path)
    print(f"\nAll plots successfully saved to {output_pdf_path}")
    plt.show()


# --------------------------------------------------------- 
# 3. NEW WRAPPER FOR PANDAS DATAFRAME
# --------------------------------------------------------- 



def fit_all_channels_in_df(parquet, column_to_fit="peak_integral", new_path=None, file_id=None, **kwargs):
    """
    Iterates through every unique channel in the dataframe, extracts the data,
    and runs the physics fit on it.
    
    If new_path is provided, it will collect all channel histograms onto a 
    single canvas (max 10 per row) and save it as 'all_channels_canvas.pdf' in that directory.
    
    Returns a Pandas DataFrame of the optimized parameters for each channel.
    """
    fit_results = {}
    df = pd.read_parquet(parquet)
    
    # Find all unique channels in the DataFrame
    channels = df['channel'].unique()
    num_plots = len(channels)
    
    # ==========================================
    # Setup the single canvas grid if plotting
    # ==========================================
    axes = None
    fig = None
    if new_path is not None and num_plots > 0:
        cols = min(10, num_plots)
        rows = math.ceil(num_plots / 10)
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 3.5 * rows))
        
        # Flatten axes array for easy 1D iteration
        if num_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

    # ==========================================
    # Iterate and Fit
    # ==========================================
    for idx, ch in enumerate(channels):
        print(f"\n=====================================")
        print(f" STARTING ANALYSIS FOR: {ch}")
        print(f"=====================================")
        
        channel_data = df[df['channel'] == ch][column_to_fit]
        
        # Assign the specific axis if we are plotting, otherwise None
        current_ax = axes[idx] if axes is not None else None
        
        # Capture all THREE return variables from the updated fit_pandas_data
        popt, pcov, peak_params = fit_pandas_data(
            channel_data, 
            channel_name=ch, 
            ax=current_ax, 
            **kwargs
        )
        
        if popt is not None:
            # Store the baseline properties 
            channel_dict = {
                "base_mu_0": popt[2],       
                "gain": popt[4],            
                "base_sigma_0": popt[3],    
                "base_sigma_1": popt[5],    
                "raw_popt": popt,
                "raw_pcov": pcov
            }
            
            # Dynamically unpack every peak and save its mu and sigma
            for n, (mu, sigma) in enumerate(peak_params):
                channel_dict[f"mu_{n}"] = mu
                channel_dict[f"sigma_{n}"] = sigma
                
            fit_results[ch] = channel_dict
            
    # ==========================================
    # Finalize and Save the Canvas
    # ==========================================
    if new_path is not None and num_plots > 0:
        # Hide any unused axes if num_plots isn't a perfect multiple of 10
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        # Create directory if it doesn't exist, then save
        os.makedirs(new_path, exist_ok=True)
        save_file = os.path.join(new_path, "all_channels_canvas.pdf")
        plt.savefig(save_file)
        print(f"\nAll plots successfully saved to {save_file}")
        
        # Close the figure to free up memory (important when looping over many files)
        plt.close(fig)

    # ==========================================
    # Build DataFrame
    # ==========================================
    # Convert the dictionary of results into a Pandas DataFrame
    results_df = pd.DataFrame.from_dict(fit_results, orient='index')
    
    # Move the channel names from the index into their own column
    results_df.index.name = 'channel'
    results_df = results_df.reset_index()
    
    # If a file identifier was provided, add it as the first column
    if file_id is not None:
        results_df.insert(0, 'file_id', file_id)
        
    return results_df

#input a root file with data from several channels
#plots several randomly picket events for each channel
def plot_sample_signals_from_root(filepath, conditions, num_samples=2, new_path=None):
    """
    Plots sample signals from a DAQ ROOT file for all available channels 
    on a single grid canvas.

    Parameters:
    filepath (str): Path to the ROOT file.
    conditions (list/tuple): Experimental conditions [Temp, LED V, SiPM Bias V].
    num_samples (int): Number of sample plots to generate per channel.
    new_path (str): Optional path to save the resulting combined plot.
    """

    # 1. Open the ROOT file and extract the data
    with uproot.open(filepath) as f:
        tree = f['Events']
        # Read branches matching the channel regex into a dictionary of numpy arrays
        data_dict = tree.arrays(filter_name="/^ch[0-9]+$/", library="np")

    if not data_dict:
        print("Error: No branches matching '/^ch[0-9]+$/' were found.")
        return

    # 2. Pre-process and collect all valid signals to plot
    # This allows us to figure out the exact grid size we need
    plot_tasks = []
    for channel_name in data_dict.keys():
        print(f"Sampling {channel_name}...")
        samples_matrix = data_dict[channel_name]
        
        total_rows = len(samples_matrix)
        actual_num_samples = min(num_samples, total_rows)
        
        # Pick random row indices without replacement for the current channel
        random_indices = np.random.choice(total_rows, size=actual_num_samples, replace=False)
        
        for row_idx in random_indices:
            signal = np.asarray(samples_matrix[row_idx], dtype=float)
            valid_mask = ~np.isnan(signal)
            sig = signal[valid_mask]
            
            # Skip if signal is too short
            if len(sig) < 50:
                continue
                
            plot_tasks.append((channel_name, row_idx, sig))

    total_plots = len(plot_tasks)
    if total_plots == 0:
        print("No valid signals found to plot.")
        return

    # 3. Setup the subplot grid (max 10 columns)
    max_cols = 10
    ncols = min(max_cols, total_plots)
    nrows = int(np.ceil(total_plots / ncols))

    # Adjust figsize dynamically based on grid size (e.g., 6 units wide, 4 units tall per plot)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4.5), squeeze=False)
    axes = axes.flatten() # Flatten to easily iterate over a 1D array

    # 4. Iterate over collected tasks and plot them on their respective axes
    for i, (channel_name, row_idx, sig) in enumerate(plot_tasks):
        ax = axes[i]
        t = np.arange(len(sig))

        # Calculate baseline and sigma
        bkg_start = baseline_start_64
        bkg_end = baseline_end_64
        bkg_sig = sig[bkg_start:bkg_end]
        baseline = np.mean(bkg_sig)
        sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        # Find peak window indices
        start_indx = peak_start_64
        end_indx = peak_end_64

        # Safety check to ensure window indices are within bounds
        start_indx = max(0, start_indx)
        end_indx = min(len(t), end_indx)

        t_window = t[start_indx:end_indx]
        sig_window = sig[start_indx:end_indx]

        # Plotting onto the specific axis
        ax.plot(t, sig, label=f'Signal ({channel_name})', color='blue')
        ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
        ax.axhline(y=baseline + sigma, color='orange', linestyle='--', label='+/- 1 Sigma')
        ax.axhline(y=baseline - sigma, color='orange', linestyle='--')
        
        if 0 <= start_indx < len(t):
            ax.axvline(x=t[start_indx], color='black', linestyle='--', label='Peak window')
        if 0 <= end_indx - 1 < len(t):
            ax.axvline(x=t[end_indx - 1], color='black', linestyle='--')

        # Color the area between the baseline and the peak
        if len(t_window) > 0:
            max_val = np.max(sig_window)
            min_val = np.min(sig_window)
            
            if abs(max_val - baseline) >= abs(min_val - baseline):
                fill_condition = (sig_window > baseline)
            else:
                fill_condition = (sig_window < baseline)

            ax.fill_between(t_window, sig_window, baseline, 
                             where=fill_condition, 
                             color='purple', alpha=0.3, interpolate=True, label='Peak Area')
     
        # Titles and labels
        ax.set_title(
            f'Sample Signal {i+1} (Row: {row_idx}, Ch: {channel_name})\n'
            f'{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
            f'LED: {float(conditions[1]):.1f} V | SiPM: {float(conditions[2]):.1f} V',
            fontsize=10
        )
        ax.set_xlabel('Samples')
        ax.set_ylabel('Signal (ADC)')
        ax.legend(fontsize=8)
        ax.grid(True)

    # 5. Clean up any unused subplots (if total_plots isn't a perfect multiple of max_cols)
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    # Tight layout prevents overlapping text
    plt.tight_layout()

    # 6. Display/Save the combined figure
    if new_path is not None:
        original_folder_name = Path(filepath).parent.name
        save_dir = Path(new_path) / original_folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Define a single filename for the canvas
        plot_filename = "combined_sample_signals.pdf"
        full_save_path = save_dir / plot_filename
        
        plt.savefig(full_save_path, dpi=200) # Added explicit DPI to keep big images sharp
        print(f"Combined plot saved to: {full_save_path}")

    plt.show() # Display the single big canvas

    return save_dir if new_path is not None else None
                


def plot_waveform_correlations(parquet, new_path=None, channel=None, plot_type='hexbin'):
    """
    Plots baseline vs area, baseline vs sigma, and sigma vs area.
    Creates a grid where each row represents one channel.
    """
    df = pd.read_parquet(parquet)
    
    # 1. Determine which channels we are plotting
    if channel:
        channels_to_plot = [channel]
    else:
        # Sort them so ch01, ch02, etc. appear in order top-to-bottom
        channels_to_plot = sorted(df['channel'].unique())  

    num_channels = len(channels_to_plot)

    # 2. Create a grid: N rows (channels) x 3 columns (plot types)
    # The height dynamically scales (5 inches per channel) so it doesn't get squished
    # squeeze=False ensures 'axes' is always a 2D array, even if there's only 1 channel
    fig, axes = plt.subplots(nrows=num_channels, ncols=3, 
                             figsize=(18, 5 * num_channels), 
                             squeeze=False)
    
    plots = [
        {'x': 'baseline', 'y': 'peak_integral', 'xlab': 'Baseline', 'ylab': 'Area (Peak Integral)', 'title': 'Baseline vs Area'},
        {'x': 'baseline', 'y': 'std_dev', 'xlab': 'Baseline', 'ylab': 'Sigma (Std Dev)', 'title': 'Baseline vs Sigma'},
        {'x': 'std_dev', 'y': 'peak_integral', 'xlab': 'Sigma (Std Dev)', 'ylab': 'Area (Peak Integral)', 'title': 'Sigma vs Area'}
    ]

    # 3. Loop over the channels (rows)
    for row_idx, ch in enumerate(channels_to_plot):
        plot_df = df[df['channel'] == ch]
        
        # Loop over the 3 plots (columns)
        for col_idx, p in enumerate(plots):
            ax = axes[row_idx, col_idx] # Select the specific subplot in the grid
            
            if plot_type == 'hexbin':
                # Add rasterized=True to the hexbin arguments
                hb = ax.hexbin(plot_df[p['x']], plot_df[p['y']], gridsize=50, cmap='viridis', mincnt=1, rasterized=True)
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label('Counts')
            else:
                ax.scatter(plot_df[p['x']], plot_df[p['y']], alpha=0.3, s=10)

            ax.set_xlabel(p['xlab'], fontsize=11)
            ax.set_ylabel(p['ylab'], fontsize=11)
            
            # Put the channel name in the title so it's clear which row is which
            ax.set_title(f"[{ch}] {p['title']}", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)

    # Automatically adjust spacing so titles/labels don't overlap
    plt.tight_layout()

    # 4. Save the single large grid
    if new_path is not None:
        save_dir = Path(new_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Name the file dynamically based on whether it's one channel or all
        if channel:
            plot_filename = f"heatmaps_{channel}.pdf"
        else:
            plot_filename = "heatmaps_all_channels.pdf"
            
        full_save_path = save_dir / plot_filename
        
        plt.savefig(full_save_path, dpi=200, bbox_inches='tight') 
        print(f"Heatmaps saved to: {full_save_path}")
        
    plt.show()




def master_function_root(filepath):

    #Analse the root file and save the data to a parquet file, return the data and the parquet file path
    data, parquet_file = analyse_root_data(filepath,parquet_folder)

    #extract the metadata from the root file and use it to label the plots
    metadata = parse_folder_metadata(filepath)
    conditions = [metadata['temp_c'], metadata['led_voltage'], metadata['bias_voltage']]

    #plot randomly selected signals from the root file for each channel, and save the plots to a folder, return save folder path
    save_dir = plot_sample_signals_from_root(filepath, conditions = conditions, num_samples=2, new_path = plot_folder)

    #fit the data for each channel obtained from the parquet file
    fit_df = fit_all_channels_in_df(parquet_file ,new_path = save_dir, column_to_fit="peak_integral",max_reduced_chi2= 100)

    #plot the correlations between baseline, sigma and area for each channel, and save the plots to a folder
    plot_waveform_correlations(parquet_file,new_path = save_dir, channel=None, plot_type='hexbin')









####################################################################
# CSV DATA
####################################################################

#Inputs csv generated from Davids Oscilloscope, with the first two columns being time and signal.
def compute_photon_signal_single_peak(filepath):
    """
    Looks for the photon peak in the signal, extracts it and integrates the
    signal above a 3 sigma threshold.

    Parameters:
    filepath (str): The path to the file containing the photon signal data.
                    A csv file which has x and y values on the first two columns.

    Returns:
    float, float, float: area, sigma, bkg average value
    """
    ###############################################################################################
    # Actual implementation
    ###############################################################################################

    df = pd.read_csv(filepath)
    df = df.rename(columns={'in s': 'time_s', 'C1 in V': 'signal_v'})

    peak = df['signal_v'].max()
    peak_time = df.loc[df['signal_v'] == peak, 'time_s'].iloc[0]
    total_avg = df['signal_v'].mean()  # abritrarly selected baseline

    # search the point where the signal reaches the total_avg after the maximum peak
    post_peak_df = df[(df['time_s'] >= peak_time) & (df['signal_v'] <= total_avg)]
    if not post_peak_df.empty:
        # get the first point after the peak that drops to baseline
        peak_end = post_peak_df.iloc[0]

    else:
    #don't save anything and skip it entirely
        return np.nan


    # do the same for the pre peak point earlier in time
    pre_peak_df = df[(df['time_s'] <= peak_time) & (df['signal_v'] <= total_avg)]

    if not pre_peak_df.empty:
        # get the last point before the peak that was at baseline
        peak_start = pre_peak_df.iloc[-1]

    else:
    #don't save anything and skip it entirely
        return np.nan



    # create a mask that removes the signal
    pre_peak_bkg = (df['time_s'] < peak_start['time_s'])
    post_peak_bkg = (df['time_s'] > peak_end['time_s'])

    # call it background
    background_df = df[pre_peak_bkg]
    background_df = background_df.rename(columns={'signal_v': 'pre_peak_bkg_v'})

    # save the rest of the data as signal
    signal_df = df[~(post_peak_bkg & pre_peak_bkg)]

    # average the background signal

    background_avg = background_df['pre_peak_bkg_v'].mean()

    # compute one sigma deviation of the background_avg
    sigma = background_df['pre_peak_bkg_v'].std()

    significant_signals = (signal_df['signal_v'] > background_avg + 3 * sigma)
    signal_above_sigma = signal_df[significant_signals]
    area = np.trapezoid(signal_above_sigma['signal_v'] - (background_avg + 3 * sigma), signal_above_sigma['time_s'])

    return area, sigma, background_avg









#computes area of peak, baseline level and sigma of baseline
#input is the file obtained from the 4ch desktop digitizer
def compute_DAQ_data(filepath):
    """
    Extracts photon signals.
    
    Parameters:
    filepath (str): Path to the CSV file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        original_keys = f.readline().strip().split(';')
        if 'SAMPLES' in original_keys:
            original_keys[original_keys.index('SAMPLES')] = 'SAMPLES1'
        max_cols = max(len(line.strip().split(';')) for line in f)

    if max_cols > len(original_keys):
        extra_keys_needed = max_cols - len(original_keys)
        for i in range(extra_keys_needed):
            original_keys.append(f"SAMPLES{i+2}")

    df = pd.read_csv(filepath, sep=';', names=original_keys, skiprows=1, index_col=False)
    
    
    samples_matrix = df.filter(like='SAMPLES').to_numpy()
    master_time = np.arange(samples_matrix.shape[1])

    array_area = []
    array_sigma = []
    array_bkg = []


    total_events = 0



    for row_index, signal in enumerate(samples_matrix):
        total_events += 1 #delete afterwards

        valid_mask = ~np.isnan(signal)
        analysis_sig = signal[valid_mask]
        t = master_time[valid_mask]
        
        if len(analysis_sig) < 50: 
            continue

        # 2. Calculate True Baseline (Assuming first 10% of waveform is pre-trigger)
        bkg_end = max(10, int(len(analysis_sig) * 0.1))
        bkg_sig = analysis_sig[:bkg_end]
        
        baseline = np.mean(bkg_sig)
        baseline_sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        
        # 1-Photon Event
        peak_idx = len(t)//2+43
        
        # A. Calculate a LOCAL baseline right before the peak 
        # This compensates for the slow, wavy drift in your raw data
        bkg_start = 0
        bkg_end = max(1, peak_idx - 35)
        local_bkg_sig = analysis_sig[bkg_start:bkg_end]
        
        if len(local_bkg_sig) > 0:
            local_baseline = np.mean(local_bkg_sig)
        else:
            local_baseline = baseline

        # B. Find Pre-Peak Intersection 35 samples before peak
        pre_peak_idx = max(0, peak_idx - 35)

        # C. Find Post-Peak Intersection 60 samples after peak
        post_peak_idx = peak_idx + 60

        # 5. Extract and Integrate 
        signal_window = analysis_sig[pre_peak_idx : post_peak_idx]
        time_window = t[pre_peak_idx : post_peak_idx]

        if len(signal_window) < 2:
            continue

        # Subtract the local baseline for accurate charge calculation
        area = np.trapezoid(signal_window - local_baseline, time_window)

        array_area.append(area)
        array_sigma.append(baseline_sigma)
        array_bkg.append(local_baseline)


    print(f"Total events: {total_events}")
    print(f"Successfully integrated: {len(array_area)}")

    return array_area, array_sigma, array_bkg



#processes files in chunks
#data obtained from the big digitizer (64ch)

def process_csv_file_in_chunks_pandas(filepath,parquet_name, chunk_size=1000):
    # Optional: keep track of baselines if you want to plot them later
    integrated_areas  = []
    baseline_averages = []
    sigma_averages    = []
    channel_ids        = []

    peak_indx  = 580
    start_indx = 560
    end_indx   = 650

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        
        # 1. Grab the event ID and channel id 
        event_id = chunk['event'].iloc[0]
        channel_id = chunk['channel'].iloc[0]
        channel_ids.append(channel_id)

        if chunk['event'].nunique() > 1:
            raise ValueError(f"Chunk contains multiple events! Chunk started with event {event_id}.")
        
        # 2. Filter the chunk for just the baseline samples (0 to 560)
        baseline_region = chunk[(chunk['sample'] >= 0) & (chunk['sample'] <= start_indx)]
        
        # 3. Compute the mean of the 'adc' column for this specific event
        event_baseline = baseline_region['adc'].mean()
        baseline_averages.append(event_baseline)

        # 4. Compute the standard deviation (sigma) of the CURRENT event's baseline
        sigma = baseline_region['adc'].std()
        sigma_averages.append(sigma)

        # 5. Find the signal window
        signal_window = chunk[(chunk['sample'] >= start_indx) & (chunk['sample'] <= end_indx)]

        # 6. Calculate the deviation from the baseline
        deviation = signal_window['adc'] - event_baseline

        # 7. Prepare values for integration
        # np.abs() handles both negative and positive peaks by looking at magnitude.
        # If the magnitude of the pulse is > 3*sigma, keep it; otherwise set to 0.
        y_vals = np.abs(deviation)
        x_vals = signal_window['sample']

        # 8. Integrate the signal in this region
        area = np.trapezoid(y_vals, x=x_vals)
        integrated_areas.append(area)   

    result_df = pd.DataFrame({
        'channel': channel_ids,
        'areas'  : integrated_areas,
        'sigmas' : sigma_averages,
        'baseline': baseline_averages
    }) 

    new_path = f'/disk/gfs_atp/hekinc/Master_measurements/{parquet_name}.parquet'      

    result_df.to_parquet(new_path,index=False)
    return result_df












def plotter(array_area, array_sigma, array_bkg, bins=500):
    """
    Plots the results of the photon signal analysis.

    Parameters:
    array_area (list): List of integrated areas for each event.
    array_sigma (list): List of baseline noise (sigma) for each event.
    array_bkg (list): List of baseline averages for each event.
    """

    # 1. Filter out any NaNs from all arrays just to be safe
    clean_area = [a for a in array_area if not np.isnan(a)]
    clean_sigma = [s for s in array_sigma if not np.isnan(s)]
    clean_bkg = [b for b in array_bkg if not np.isnan(b)]

    # 2. Automatically calculate the "core" range for Sigma and Bkg 
    # This finds the boundaries that contain 99% of your data, ignoring massive outliers
    sigma_min, sigma_max = np.percentile(clean_sigma, [0, 99])
    bkg_min, bkg_max = np.percentile(clean_bkg, [1, 99])

    # 3. Create a figure with 3 subplots in a single row
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # --- Plot 1: Area (Charge Spectrum) ---
    axs[0, 0].hist(clean_area, bins=bins, color='blue', alpha=0.7)
    axs[0, 0].set_title('Histogram of Area (Core)')
    axs[0, 0].set_xlabel('Area')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].grid(True, alpha=0.3)

    # --- Plot 2: Sigma (Baseline Noise) ---
    axs[0, 1].hist(clean_sigma, bins=200, range=(sigma_min, sigma_max), color='green', alpha=0.7)
    axs[0, 1].set_title('Histogram of Sigma (Noise)')
    axs[0, 1].set_xlabel('Sigma')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True, alpha=0.3)

    # --- Plot 3: Background Baseline ---
    axs[0, 2].hist(clean_bkg, bins=200, range=(bkg_min, bkg_max), color='red', alpha=0.7)
    axs[0, 2].set_title('Histogram of Background Baseline')
    axs[0, 2].set_xlabel('Baseline Level')
    axs[0, 2].set_ylabel('Frequency')
    axs[0, 2].grid(True, alpha=0.3)


    axs[1,0].plot(clean_area, clean_sigma, 'o', markersize=2, alpha=0.5)
    axs[1,0].set_title('Sigma vs Area')
    axs[1,0].set_xlabel('Area')
    axs[1,0].set_ylabel('Sigma')
    axs[1,0].grid(True, alpha=0.3)

    axs[1,1].plot(clean_area, clean_bkg, 'o', markersize=2, alpha=0.5)
    axs[1,1].set_title('Background Baseline vs Area')
    axs[1,1].set_xlabel('Area')
    axs[1,1].set_ylabel('Background Baseline')
    axs[1,1].grid(True, alpha=0.3)

    axs[1,2].plot(clean_sigma, clean_bkg, 'o', markersize=2, alpha=0.5)
    axs[1,2].set_title('Background Baseline vs Sigma')
    axs[1,2].set_xlabel('Sigma')
    axs[1,2].set_ylabel('Background Baseline')
    axs[1,2].grid(True, alpha=0.3)

    # Adjust spacing so labels don't overlap and show the plots
    plt.tight_layout()
    plt.show()











# --------------------------------------------------------- 
# 2. Main Analysis and Fitting Function 
# --------------------------------------------------------- 

def analyze_and_fit_physics(raw_data, bins=500, num_peaks_to_fit=18, 
                            confident_peak_limit=7, manual_mu0=None, manual_gain=None): 
    """ 
    Fits physical data using a constrained multi-photon physics model. 
    If manual_mu0 and manual_gain are provided, the automatic peak detector is bypassed. 
    """ 
    raw_data = np.array(raw_data, dtype=float) 
    print(f"Processing {len(raw_data)} data points...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 

    # ========================================== 
    # Step A & B: Find Anchors (Manual or Auto) 
    # ========================================== 
    if manual_mu0 is not None and manual_gain is not None: 
        print("Using manual physics overrides for Pedestal and Gain.") 
        mu_0_guess = manual_mu0 
        gain_guess = manual_gain 
    else: 
        print("Auto-detecting peaks...") 
        smoothed_counts = gaussian_filter1d(counts, sigma=1) 
        peaks, _ = find_peaks( 
            smoothed_counts,  
            prominence=np.max(smoothed_counts) * 0.01,  
            distance=4 
        ) 
        
        # Fail-safe to ensure we at least capture the pedestal 
        global_max_bin = np.argmax(counts) 
        if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
            peaks = np.append(peaks, global_max_bin) 
            peaks = np.sort(peaks) 

        if len(peaks) == 0: 
            print("Failed to find any peaks. Try providing manual_mu0 and manual_gain.") 
            return None, None 

        idx_0 = peaks[0] 
        mu_0_guess = bin_centers[idx_0] 
        
        # Estimate gain (distance to next peak). If no 2nd peak, guess based on range. 
        if len(peaks) > 1: 
            gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
        else: 
            gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

    # Estimate widths 
    sigma_0_guess = max(dx, 1e-9) 
    sigma_1_guess = sigma_0_guess * 0.5  

    # ========================================== 
    # Step C: Build Physics-Based Guesses & Bounds 
    # ========================================== 
    p0 = [] 
    lower_bounds = [] 
    upper_bounds = [] 

    # 1. Background Guesses 
    A_bg_guess = np.max(counts) * 0.1 
    lambda_bg_guess = 1.0 / max(np.mean(raw_data), 1e-9) 
    p0.extend([A_bg_guess, lambda_bg_guess]) 
    
    lower_bounds.extend([0, 0]) 
    upper_bounds.extend([np.max(counts) * 0.5, 5000]) # Strict background ceiling 

    # 2. Physics Parameter Guesses (mu_0, sigma_0, gain, sigma_1) 
    p0.extend([mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
    
    # BOUNDS: Force sigma_0 and sigma_1 to stay under 40% of the gain, preserving the valleys! 
    lower_bounds.extend([bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
    upper_bounds.extend([bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

    # 3. Amplitude Guesses for ALL peaks (0 through num_peaks_to_fit) 
    for n in range(num_peaks_to_fit): 
        expected_mu = mu_0_guess + (n * gain_guess) 
        
        # Guess the amplitude by looking at the histogram height at that x-value 
        if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
            closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
            A_guess = counts[closest_bin_idx] 
        else: 
            A_guess = 1e-9  
            
        p0.append(A_guess) 
        lower_bounds.append(0.0) 
        upper_bounds.append(np.inf) 

    # Force guesses strictly inside bounds 
    p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

    # ========================================== 
    # Step D: Fit the curve 
    # ========================================== 
    try: 
        popt, pcov = curve_fit( 
            fit_model_physics,  
            bin_centers,  
            counts,  
            p0=p0,  
            bounds=(lower_bounds, upper_bounds), 
            maxfev=25000  
        ) 
        print("Physics Curve fitting converged successfully!") 

        expected_counts = fit_model_physics(bin_centers, *popt)
            
        # Only calculate chi-square for bins with at least 1 count to avoid dividing by zero
        valid_bins = counts > 0
        O_i = counts[valid_bins]
        E_i = expected_counts[valid_bins]
        
        # Calculate Chi-Square
        chi_square = np.sum(((O_i - E_i) ** 2) / O_i)
        
        # Calculate Degrees of Freedom (Number of data points - Number of fitted parameters)
        ndf = len(O_i) - len(popt)
        
        # Calculate Reduced Chi-Square
        reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
        # -----------------------------------

    except Exception as e: 
        print(f"Optimal parameters not found: {e}") 
        return None, None 

    # ========================================== 
    # Step E: Plotting and Output 
    # ========================================== 
    plt.figure(figsize=(10, 6)) 
    plt.hist(raw_data, bins=bins, alpha=0.5, color='blue', label='Binned Data') 
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
    
    # Plot Total Fit 
    plt.plot(x_fit, fit_model_physics(x_fit, *popt), color='red', lw=2, label='Total Fit') 
    
    # Extract optimized physics parameters 
    A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1 = popt[0:6] 
    amplitudes = popt[6:] 
    
    # Plot Exponential Background 
    #plt.plot(x_fit, A_bg * np.exp(-lambda_bg * x_fit), color='black', linestyle=':', lw=2, label='Exp Background') 
    
    print("\n--- Physical Fit Results ---") 
    print(f"Pedestal (\u03BC_0): {mu_0:.6e}") 
    print(f"Detector Gain:      {gain:.6e}") 
    print(f"Pedestal Noise (\u03C3_0): {sigma_0:.6e}") 
    print(f"1-Photon Noise (\u03C3_1): {sigma_1:.6e}") 
    print(f"\u03C7\u00B2 / NDF:           {chi_square:.2f} / {ndf} = {reduced_chi_square:.3f}")
    print("-" * 28) 
    
    tail_sum = np.zeros_like(x_fit) 
    
    for n, A in enumerate(amplitudes): 
        mu_n = mu_0 + n * gain 
        sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2) 
        single_gauss = A * np.exp(-0.5 * ((x_fit - mu_n) / sigma_n)**2) 
        
        # Only plot and print individual peaks up to the limit 
        if n <= confident_peak_limit: 
            # Check if amplitude is large enough to bother plotting 
            if np.max(single_gauss) > (np.max(counts) * 0.005): 
                plt.plot(x_fit, single_gauss, '--', label=f'{n}-Photon') 
                print(f"Peak {n}: \u03BC = {mu_n:.6e}, \u03C3 = {sigma_n:.6e}") 
        else: 
            # Group the higher-order peaks into the tail 
            tail_sum += single_gauss 

    # Plot the aggregated High-Photon Tail if it exists 
    if np.max(tail_sum) > (np.max(counts) * 0.005): 
        plt.plot(x_fit, tail_sum, color='gray', linestyle='-.', lw=1.5, label='High-Photon Tail') 

    plt.xlabel('ADC Counts') 
    plt.ylabel('Frequency') 
    plt.title('Histogramm of Area') 
    
    # Move legend outside the plot if it gets too crowded 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(alpha=0.3) 
    plt.tight_layout() 
    plt.show() 

    return popt, pcov






#function that takes the filepath and frequency and returns 10 sample plots of the signal with the peak, baseline and sigma marked on the plot. The plots should be saved in a folder called "plots" in the current working directory.
def plot_sample_signals(filepath,conditions, num_samples=10, ):
    """
    Plots sample signals from the DAQ data with peak, baseline, and sigma marked.

    Parameters:
    filepath (str): Path to the CSV file.
    conditions (list): Experimental conditions [Temp, LED V, SiPM Bias V].
    num_samples (int): Number of sample plots to generate.
    """
   

    # Create a directory for plots if it doesn't exist
    #if not os.path.exists('plots'):
    #    os.makedirs('plots')

    with open(filepath, 'r', encoding='utf-8') as f:
        original_keys = f.readline().strip().split(';')
        #rename the first SAMPLES key to SAMPLES1 to avoid duplicate column names
        original_keys[original_keys.index('SAMPLES')] = 'SAMPLES1'
        
        # Scan the rest of the file to find the longest row
        max_cols = max(len(line.strip().split(';')) for line in f)

    # Generate new keys to fill the gap (e.g., if max_cols is 10 and original is 4)
    if max_cols > len(original_keys):
        extra_keys_needed = max_cols - len(original_keys)
        for i in range(extra_keys_needed):
            original_keys.append(f"SAMPLES{i+2}")

    # Pass 2: Let Pandas read the whole file using your newly extended keys
    df = pd.read_csv(
        filepath, 
        sep=';', 
        names=original_keys, 
        skiprows=1,           # Skip the original broken header
        index_col=False
    )

    # Extract data to numpy array
    samples_matrix = df.filter(like='SAMPLES').to_numpy()
    master_time = np.arange(samples_matrix.shape[1])

    # 1. Determine how many samples we can actually pick
    total_rows = samples_matrix.shape[0]
    actual_num_samples = min(num_samples, total_rows)

    # 2. Pick random row indices without replacement
    random_indices = np.random.choice(total_rows, size=actual_num_samples, replace=False)

    # 3. Iterate over the randomly selected indices
    for i, row_idx in enumerate(random_indices):
        signal = samples_matrix[row_idx]   # <-- Grab the randomly selected row
        valid_mask = ~np.isnan(signal)
        sig = signal[valid_mask]
        t = master_time[valid_mask]

        if len(sig) < 50: 
            continue

        # Calculate baseline and sigma
        bkg_end = max(10, int(len(sig) * 0.1))
        bkg_sig = sig[:bkg_end]
        baseline = np.mean(bkg_sig)
        sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        # Find peak
        peak_idx = len(t)//2+43
        start_indx = peak_idx - 35
        end_indx = peak_idx + 60

        t_window = t[start_indx:end_indx]
        sig_window = sig[start_indx:end_indx]

        
        #peak_value = sig[peak_idx]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t, sig, label='Signal', color='blue')
        plt.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
        plt.axhline(y=baseline +sigma, color='orange', linestyle='--', label='1 Sigma Threshold')
        plt.axvline(x=t[peak_idx], color='red', linestyle='--', label='Peak')
        plt.axvline(x=t[start_indx], color='black', linestyle='--', label='Peak-35 Samples')
        plt.axvline(x=t[end_indx], color='black', linestyle='--', label='Peak+60 Samples')

        #colour the area between three sigma line and the peak
        
        plt.fill_between(t_window, sig_window, baseline, 
                         where=(sig_window > baseline), 
                         color='purple', alpha=0.3, interpolate=True, label='Peak Area')
     
        
        # It's helpful to include the actual row index in the title and filename
        plt.title(
        f'Sample Signal {i+1} (Original Row: {row_idx})\n'
        f'Data taken at {float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
        f'LED: {float(conditions[1]):.1f} V | SiPM Bias: {float(conditions[2]):.1f} V'
        )
        plt.xlabel('Samples')

        plt.ylabel('Signal (ADC)')
        plt.legend()
        plt.grid()
        
        # Save the plot
        plt.show()







def parse_folder_metadata(filepath):
    """
    Extracts experimental parameters from the parent folder's name.
    Example filepath: .../20260814T095731Z-single-ch-100c-2-3-54v-400hz-10ns-3v-5192c7fd/data.csv
    """
    # Grab the directory path, then get the final folder name from that path
    folder_name = os.path.basename(os.path.dirname(filepath))
    
    # Initialize a dictionary with the raw folder name
    metadata = {"raw_folder_name": folder_name}
    
    # 1. Extract Timestamp
    timestamp_match = re.match(r"^(\d{8}T\d{6}Z)", folder_name)
    if timestamp_match:
        metadata["timestamp"] = timestamp_match.group(1)
        
    # 2. Extract Channel Type
    ch_match = re.search(r"(single-ch|multi-ch)", folder_name)
    if ch_match:
        metadata["channel_type"] = ch_match.group(1)
        
    # 3. Extract voltages (e.g., 54v, 3v)
    voltages = re.findall(r"-(\d+)v", folder_name)
    if len(voltages) >= 1:
        metadata["bias_voltage"] = float(voltages[0])  
    if len(voltages) >= 2:
        metadata["pulse_voltage"] = float(voltages[1]) 
        
    # 4. Extract Frequency (e.g., 400hz)
    freq_match = re.search(r"[-_](\d+)[hH][zZ]", folder_name)
    if freq_match:
        metadata["frequency_hz"] = float(freq_match.group(1))
        
    # 5. Extract Time window/pulse width (e.g., 10ns)
    time_match = re.search(r"-(\d+)ns", folder_name)
    if time_match:
        metadata["time_ns"] = float(time_match.group(1))
        
    # 6. Extract Temp/Condition (e.g., 100c)
    temp_match = re.search(r"-(\d+)c-", folder_name)
    if temp_match:
        metadata["temp_c"] = float(temp_match.group(1))
        
    # 7. Extract the unique trailing hash (e.g., 5192c7fd)
    hash_match = re.search(r"-([a-f0-9]{8})$", folder_name)
    if hash_match:
        metadata["run_hash"] = hash_match.group(1)

    # 8. Extract the LED voltage 
    all_voltages = re.findall(r"[-_]([0-9]+(?:[-.][0-9]+)?)[vV]", folder_name)
    
    if all_voltages:
        # Grab the last match found in the string (e.g., "3-4")
        last_voltage_str = all_voltages[-1]
        
        # Replace the hyphen with a dot so Python can read it as a decimal
        last_voltage_str = last_voltage_str.replace("-", ".")
        
        metadata["led_voltage"] = float(last_voltage_str)


    return metadata


















################################################################################
#ARCHIVES 
################################################################################
'''
def fit_model_physics(x, *params): 
    """ 
    Physics-linked model for SiPM/PMT multi-photon spectra. 
    params[0]: A_bg       (Exponential background amplitude) 
    params[1]: lambda_bg  (Exponential background decay rate) 
    params[2]: mu_0       (Position of the 0-photon pedestal) 
    params[3]: sigma_0    (Width of the 0-photon pedestal) 
    params[4]: gain       (Distance between adjacent photon peaks) 
    params[5]: sigma_1    (Additional width contribution per photon) 
    params[6:]: A_0, A_1, A_2... (Amplitudes for each photon peak) 
    """ 
    A_bg = params[0] 
    lambda_bg = params[1] 
    mu_0 = params[2] 
    sigma_0 = params[3] 
    gain = params[4] 
    sigma_1 = params[5] 
    amplitudes = params[6:] 
    
    # Calculate Exponential Background 
    y = A_bg * np.exp(-lambda_bg * x) 
    
    # Add Physics-Linked Gaussians 
    for n, A in enumerate(amplitudes): 
        # Position: Pedestal + (n * Gain) 
        mu_n = mu_0 + n * gain 
        
        # Width: Add standard deviations in quadrature (Poisson statistics) 
        sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2) 
        
        # Add the specific Gaussian to the total line 
        y += A * np.exp(-0.5 * ((x - mu_n) / sigma_n)**2) 
        
    return y

#fitting function for dataframe obtained from root file

def fit_pandas_data(raw_data, channel_name="Data", bins=500, num_peaks_to_fit=18, 
                    confident_peak_limit=7, manual_mu0=None, manual_gain=None, ax=None,
                    max_reduced_chi2= None): 
    
    # Drop NaN values that might exist in Pandas and convert to float array
    raw_data = np.array(raw_data.dropna(), dtype=float) 
    print(f"Processing {len(raw_data)} data points for {channel_name}...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 
    
    # Initialize return variables
    fit_successful = False
    popt, pcov = None, None
    peak_params = [] # Array to hold (mu, sigma) for each peak
    
    # Initialize chi-square variables so they exist for the plotting step
    reduced_chi_square = np.nan
    chi_square = np.nan
    ndf = 0

    # ========================================== 
    # Step A & B: Find Anchors 
    # ========================================== 
    if manual_mu0 is not None and manual_gain is not None: 
        print("Using manual physics overrides for Pedestal and Gain.") 
        mu_0_guess = manual_mu0 
        gain_guess = manual_gain 
    else: 
        print("Auto-detecting peaks...") 
        smoothed_counts = gaussian_filter1d(counts, sigma=1) 
        peaks, _ = find_peaks( 
            smoothed_counts,  
            prominence=np.max(smoothed_counts) * 0.01,  
            distance=4 
        ) 
        
        global_max_bin = np.argmax(counts) 
        if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
            peaks = np.append(peaks, global_max_bin) 
            peaks = np.sort(peaks) 

        if len(peaks) == 0: 
            print("Failed to find any peaks. Skipping fit and plotting raw data.")
        else:
            idx_0 = peaks[0] 
            mu_0_guess = bin_centers[idx_0] 
            
            if len(peaks) > 1: 
                gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
            else: 
                gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

    # Only attempt to build bounds and fit if we found peaks (or used manual inputs)
    if (manual_mu0 is not None) or (len(peaks) > 0):
        sigma_0_guess = max(dx, 1e-9) 
        sigma_1_guess = sigma_0_guess * 0.5  

        # ========================================== 
        # Step C: Build Bounds 
        # ========================================== 
        p0 = [] 
        lower_bounds = [] 
        upper_bounds = [] 

        A_bg_guess = np.max(counts) * 0.1 
        lambda_bg_guess = 1.0 / max(np.mean(raw_data), 1e-9) 
        p0.extend([A_bg_guess, lambda_bg_guess]) 
        
        lower_bounds.extend([0, 0]) 
        upper_bounds.extend([np.max(counts) * 0.5, 5000]) 

        p0.extend([mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
        lower_bounds.extend([bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
        upper_bounds.extend([bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

        for n in range(num_peaks_to_fit): 
            expected_mu = mu_0_guess + (n * gain_guess) 
            if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
                closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
                A_guess = counts[closest_bin_idx] 
            else: 
                A_guess = 1e-9  
                
            p0.append(A_guess) 
            lower_bounds.append(0.0) 
            upper_bounds.append(np.inf) 

        p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

        # ========================================== 
        # Step D: Fit the curve & Check Thresholds
        # ========================================== 
        try: 
            popt_temp, pcov_temp = curve_fit( 
                fit_model_physics,  
                bin_centers,  
                counts,  
                p0=p0,  
                bounds=(lower_bounds, upper_bounds), 
                maxfev=25000  
            ) 
            
            # Calculate chi-square on the temporary fit
            expected_counts = fit_model_physics(bin_centers, *popt_temp)
            valid_bins = counts > 0
            O_i = counts[valid_bins]
            E_i = expected_counts[valid_bins]
            
            chi_square = np.sum(((O_i - E_i) ** 2) / O_i)
            ndf = len(O_i) - len(popt_temp)
            reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
            
            # --- NEW THRESHOLD LOGIC ---
            if max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                print(f"Fit rejected: Reduced \u03C7\u00B2 ({reduced_chi_square:.2f}) > threshold ({max_reduced_chi2}).")
                # Leave fit_successful as False, popt as None
            else:
                print("Physics Curve fitting converged successfully!") 
                fit_successful = True
                popt = popt_temp
                pcov = pcov_temp
                
                A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1 = popt[0:6] 
                amplitudes = popt[6:] 
                
                for n in range(len(amplitudes)):
                    mu_n = mu_0 + n * gain
                    sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2)
                    peak_params.append([mu_n, sigma_n])
            
        except Exception as e: 
            print(f"Optimal parameters not found: {e}") 

    # ========================================== 
    # Step E: Plotting to the provided Axis 
    # ========================================== 
    if ax is not None:
        ax.hist(raw_data, bins=bins, alpha=0.5, color='blue', label='Binned Data') 
        
        if fit_successful:
            x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
            ax.plot(x_fit, fit_model_physics(x_fit, *popt), color='red', lw=2, label='Total Fit') 
            
            print(f"\n--- Physical Fit Results for {channel_name} ---") 
            for n, (mu_n, sigma_n) in enumerate(peak_params[:4]):
                print(f"  {n}-Photon Peak: \u03BC = {mu_n:.6e}, \u03C3 = {sigma_n:.6e}")
            print("-" * 50) 
            print(f"\u03C7\u00B2 / NDF:           {chi_square:.2f} / {ndf} = {reduced_chi_square:.3f}")
            print("-" * 28) 
            
            tail_sum = np.zeros_like(x_fit) 
            for n, (mu_n, sigma_n) in enumerate(peak_params): 
                A = amplitudes[n]
                single_gauss = A * np.exp(-0.5 * ((x_fit - mu_n) / sigma_n)**2) 
                
                if n <= confident_peak_limit: 
                    if np.max(single_gauss) > (np.max(counts) * 0.005): 
                        ax.plot(x_fit, single_gauss, '--', label=f'{n}-Photon') 
                else: 
                    tail_sum += single_gauss 

            if np.max(tail_sum) > (np.max(counts) * 0.005): 
                ax.plot(x_fit, tail_sum, color='gray', linestyle='-.', lw=1.5, label='High-Photon Tail') 

            ax.set_title(f'{channel_name}, with \u03C7\u00B2/NDF = {reduced_chi_square:.3f}') 
        else:
            # Check why it failed to give a helpful plot title
            if not np.isnan(reduced_chi_square) and max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                ax.set_title(f'{channel_name} (REJECTED: \u03C7\u00B2/NDF > {max_reduced_chi2})') 
            else:
                ax.set_title(f'{channel_name} (FIT FAILED)') 

        ax.set_xlabel('ADC Counts') 
        ax.set_ylabel('Frequency') 
        ax.grid(alpha=0.3) 

    return popt, pcov, np.array(peak_params)


'''