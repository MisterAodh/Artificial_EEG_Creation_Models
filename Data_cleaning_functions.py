import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import FastICA
import os
import json

# Use 'TkAgg' backend for matplotlib to ensure dynamic images work in PyCharm Professional
matplotlib.use('TkAgg')

import numpy as np


def generate_noise_signals(training_data_dir, augmented_data_dir, results_path='output_datasets/results.txt', num_noise_per_signal=3):
    true_signals = []

    # Load the true signals from the training data JSON files
    for file in os.listdir(training_data_dir):
        if file.endswith('.json'):
            with open(os.path.join(training_data_dir, file), 'r') as f:
                data = json.load(f)
                true_signals.extend(data['true_time_periods'])

    # Generate noise signals for each true signal
    noise_signals = []
    for signal in true_signals:
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        for _ in range(num_noise_per_signal):
            noise_signal = np.random.normal(signal_mean, signal_std, len(signal)).tolist()
            noise_signals.append(noise_signal)

    # Prepare the directory to save the noise signals
    if not os.path.exists(augmented_data_dir):
        os.makedirs(augmented_data_dir)

    # Save the noise signals to a new JSON file
    noise_signals_data = {
        'noise_time_periods': noise_signals
    }

    noise_signals_file = os.path.join(augmented_data_dir, 'noise_signals.json')
    with open(noise_signals_file, 'w') as f:
        json.dump(noise_signals_data, f)

    print(f"Noise signals saved to {noise_signals_file}")

    # Log the results
    with open(results_path, 'a') as f:
        f.write('-' * 30 + '\n')
        f.write('Noise Signal Generation\n')
        f.write('-' * 30 + '\n')
        f.write(f'True Signals Processed: {len(true_signals)}\n')
        f.write(f'Noise Signals Generated: {len(noise_signals)}\n')
        f.write(f'Noise Signals File: {noise_signals_file}\n')
        f.write('-' * 30 + '\n\n')

class VEPDataProcessor:
    def __init__(self, vep_file_path, stim_times_file_path, sampling_rate=512, artifact_threshold=300,
                 artifact_window=50):
        self.vep_csv = pd.read_csv(vep_file_path)
        self.stim_times = pd.read_csv(stim_times_file_path).values.flatten()
        self.sampling_rate = sampling_rate
        self.artifact_threshold = artifact_threshold
        self.artifact_window = artifact_window
        self.cleaned_eeg = None

    def remove_artifacts(self, data):
        """Remove artifacts by zeroing out sections of data around high-value points."""
        for i in range(data.shape[0]):
            artifact_indices = np.where(np.abs(data[i]) > self.artifact_threshold)[0]
            for idx in artifact_indices:
                start = max(0, idx - self.artifact_window)
                end = min(len(data[i]), idx + self.artifact_window)
                data[i, start:end] = 0
        return data

    def trim_start(self, data, trim_size=1550):
        """Trim the first 1550 points of each channel after setting artifacts to zero, and adjust timestamps accordingly."""
        percentages = []
        total_data_points = data.shape[1]
        total_zeros_overall = np.sum(data == 0)
        overall_percentage_zero = (total_zeros_overall / (total_data_points * data.shape[0])) * 100

        for i in range(data.shape[0]):
            zero_count = np.sum(data[i, :trim_size] == 0)
            percentage_zero = (zero_count / trim_size) * 100
            percentages.append(percentage_zero)
            print(f"Channel {i + 1}: {percentage_zero:.2f}% of the first {trim_size} data points are zero.")

        # Trim the first 1550 points
        trimmed_data = data[:, trim_size:]
        self.stim_times = self.stim_times - trim_size
        self.stim_times = self.stim_times[self.stim_times >= 0]  # Remove any negative timestamps

        print(f"The first {trim_size} data points have been removed from all channels.")

        # Save the results to a file
        self.save_results(trim_size, overall_percentage_zero, np.mean(percentages), 'output_datasets/')

        return trimmed_data

    def save_results(self, trim_size, overall_percentage_zero, first_trim_percentage_zero, output_dir):
        """Save the results of the artifact removal process to a text file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results_path = os.path.join(output_dir, 'results.txt')
        with open(results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write('(Artifact Removal)\n')
            f.write('-' * 30 + '\n')
            f.write(f'Percentage artifacts overall in EEG data: {overall_percentage_zero:.2f}%\n')
            f.write(f'Percentage in first {trim_size} time samples: {first_trim_percentage_zero:.2f}%\n')
            f.write(f'First {trim_size} data points removed\n')
            f.write('New stimulus times created\n')
            f.write('New data file saved to output_datasets/\n')
            f.write('-' * 30 + '\n\n\n\n')

    def save_trimmed_data(self, output_dir):
        """Save the trimmed data and corresponding VEP timestamps."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the cleaned EEG data
        cleaned_data_df = pd.DataFrame(self.cleaned_eeg.T, columns=self.vep_csv.columns)
        cleaned_data_df.to_csv(os.path.join(output_dir, 'trimmed_vep_data.csv'), index=False)

        # Save the adjusted stimulus times
        pd.DataFrame(self.stim_times, columns=['Stimulus_Times']).to_csv(
            os.path.join(output_dir, 'trimmed_vep_stim_times.csv'), index=False)
        print(f"Trimmed data and stimulus times saved to '{output_dir}'.")

    def dynamic_plot(self, data, title):
        """Dynamically view the data for a specific sensor channel."""
        window_size = 300
        update_size = 5
        data_length = data.shape[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(data[0, :window_size])
        ax.set_ylim(-1000, 1000)
        ax.set_title(title)

        def update(frame):
            start = (frame * update_size) % (data_length - window_size)
            end = start + window_size
            line.set_ydata(data[0, start:end])
            return line,

        ani = FuncAnimation(fig, update, frames=range(data_length // update_size), blit=True, interval=100)
        plt.show()


    def vep_analysis(self, output_dir='output_datasets/average_channel_vep/'):
        """Perform VEP analysis for the first 9 channels and plot in a grid."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        for i in range(9):
            epochs = self.epoch_data(self.cleaned_eeg, i, self.stim_times, 100, 150, self.artifact_threshold)
            if epochs.size > 0:
                average_signal = np.mean(epochs, axis=0)
            else:
                continue  # Skip plotting if no valid epochs
            row, col = divmod(i, 3)
            axs[row, col].plot(np.arange(-100, 150), average_signal)
            axs[row, col].set_title(f'Channel {i + 1}')
            axs[row, col].set_ylim([-150, 150])
            axs[row, col].axhline(y=0, color='red', linestyle='dotted', linewidth=1)
            # Save the plot
            fig_name = f"average_vep_channel_{i+1}.png"
            plt.savefig(os.path.join(output_dir, fig_name))

        plt.tight_layout()
        plt.show()

        # Logging
        self.log_vep_analysis(output_dir)

    def epoch_data(self, eeg_data, channel_index, trig_times, pre_samples, post_samples, threshold):
        """Epoch the data and return good epochs."""
        epochs = []
        for trig_time in trig_times:
            start = trig_time - pre_samples
            end = trig_time + post_samples
            if 0 <= start and end < eeg_data.shape[1]:
                epoch = eeg_data[channel_index, start:end]
                if np.all(np.abs(epoch) <= threshold):
                    epochs.append(epoch)
        return np.array(epochs)

    def log_vep_analysis(self, output_dir):
        """Log the results of the VEP analysis."""
        results_path = os.path.join(output_dir, '../results.txt')
        with open(results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write('(VEP Analysis)\n')
            f.write('-' * 30 + '\n')
            f.write('VEP analysis performed on cleaned and trimmed data.\n')
            f.write(f'Results saved to {output_dir}\n')
            f.write('-' * 30 + '\n\n\n\n')

    def process_and_display_data(self, sensor_index=0, output_dir='output_datasets'):
        # Display the data before trimming
        print("Displaying data before trimming...")
        self.dynamic_plot(self.vep_csv.values.T, 'Before Trimming')

        # Remove artifact-dense start and end regions and zero out remaining artifacts
        self.cleaned_eeg = self.remove_artifacts(self.vep_csv.values.T)
        print("Artifacts have been zeroed out.")

        # Trim the first 1550 points and adjust the timestamps
        trim_size = 1550
        self.cleaned_eeg = self.trim_start(self.cleaned_eeg, trim_size=trim_size)

        # Save the trimmed data and adjusted VEP timestamps
        self.save_trimmed_data(output_dir)

        # Display the data after trimming
        print("Displaying data after trimming...")
        self.dynamic_plot(self.cleaned_eeg, 'After Trimming')

        # Plot channel 3 trimmed data
        plt.figure(figsize=(10, 6))
        plt.plot(self.cleaned_eeg[2], label=f'Channel 3 - Trimmed Data')
        plt.title(f'Channel 3 - Trimmed EEG Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude (µV)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'channel_3_trimmed.png'))
        plt.show()

        # Log the results
        self.log_process_and_display_data(output_dir, trim_size, sensor_index)

    def apply_ica_and_create_datasets(self, output_dir='output_datasets/ica_datasets/'):
        """Apply ICA, visualize, and create datasets based on selected components."""

        # Prompt user to select which channel to use for ICA
        channel_index = int(input(f"Enter the channel index for ICA analysis (0 to {self.cleaned_eeg.shape[0] - 1}): "))

        # Apply ICA to the selected channel's data
        n_components = int(input("Enter the number of ICA components: "))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ICA on the selected channel
        ica = FastICA(n_components=n_components, random_state=0)
        sources = ica.fit_transform(self.cleaned_eeg[channel_index].reshape(-1, 1)).T

        # Check if the number of components is reduced and adjust accordingly
        actual_components = sources.shape[0]
        if actual_components < n_components:
            print(f"Warning: Number of components reduced to {actual_components} due to data limitations.")
            n_components = actual_components

        # Plot average VEP of the ICA components
        self.plot_ica_components(sources, n_components, channel_index)

        # Create channel-specific directory for saving datasets
        channel_output_dir = os.path.join(output_dir, f'channel_{channel_index + 1}')
        if not os.path.exists(channel_output_dir):
            os.makedirs(channel_output_dir)

        while True:
            bands_input = input(
                "Enter the frequency bands (ICA components) you want (e.g., 1,2,3) or 'zzz' to finish: ").strip()
            if bands_input.lower() == 'zzz':
                break
            bands = [int(band) - 1 for band in bands_input.split(',')]

            # Create and save the new dataset
            selected_sources = sources[bands, :]
            selected_data = ica.inverse_transform(selected_sources.T).T

            dataset_name = f"ica_components_{'_'.join(map(str, [b + 1 for b in bands]))}.csv"
            pd.DataFrame(selected_data.T).to_csv(os.path.join(channel_output_dir, dataset_name), index=False)
            print(f"Dataset {dataset_name} saved to {channel_output_dir}.")

        # Log the creation of datasets
        self.log_ica_datasets(channel_output_dir, n_components, channel_index)


    def plot_ica_components(self, sources, n_components, channel_index):
        """Plot the average VEP for each ICA component."""
        fig, axs = plt.subplots(n_components, 1, figsize=(12, n_components * 3))
        for i in range(n_components):
            axs[i].plot(sources[i], label=f'ICA Component {i + 1} - Channel {channel_index + 1}')
            axs[i].set_title(f'ICA Component {i + 1} - Channel {channel_index + 1}')
            axs[i].legend()
        plt.tight_layout()
        plt.show()

    def log_ica_datasets(self, output_dir, n_components, channel_index):
        """Log the creation of ICA-based datasets."""
        results_path = os.path.join(output_dir, '../../results.txt')
        with open(results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write('(ICA Dataset Creation)\n')
            f.write('-' * 30 + '\n')
            f.write(f'ICA applied to Channel {channel_index + 1} with {n_components} components.\n')
            f.write(f'Datasets created and saved to {output_dir}.\n')
            f.write('-' * 30 + '\n\n\n')

def log_process_and_display_data(self, output_dir, trim_size, sensor_index):
    """Log the results of the process and display data function."""
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'a') as f:
        f.write('-' * 30 + '\n')
        f.write('(Process and Display Data)\n')
        f.write('-' * 30 + '\n')
        f.write(f'Sensor Index: {sensor_index + 1}\n')
        f.write(f'Data before trimming was displayed.\n')
        f.write(f'Artifacts have been removed.\n')
        f.write(f'Trimmed the first {trim_size} data points.\n')
        f.write(f'Trimmed data and adjusted VEP timestamps have been saved to {output_dir}.\n')
        f.write(f'Data after trimming was displayed.\n')
        f.write(f'Channel 3 trimmed data plot saved as channel_3_trimmed.png.\n')
        f.write('-' * 30 + '\n\n\n\n')
