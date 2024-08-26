import os
import json
import pandas as pd
import numpy as np

class VEPtrainingdata_extract:
    def __init__(self, vep_data_location, vep_stim_times_location, output_location, channels=10):
        self.vep_data_location = vep_data_location
        self.vep_stim_times_location = vep_stim_times_location
        self.output_location = output_location  # This will still be used for saving JSON files
        self.results_path = os.path.join('output_datasets', 'results.txt')  # Log to the existing results.txt
        self.channels = channels

        # Load data from CSV files
        self.vep_csv = pd.read_csv(vep_data_location)
        self.stim_times = pd.read_csv(vep_stim_times_location)['Stimulus_Times'].values

        # Define parameters for analysis
        self.artifact_threshold = 300  # Threshold for artifact detection
        self.artifact_window = 50  # Window size around artifact to set to 0
        self.local_peak_window = 80  # Window after stim time to search for local peak
        self.response_window = 100  # Window size around the peak to capture VEP

        # Initialize logging strings
        self.log_entries = []

    def log_results(self, true_veps_counts, false_veps_counts):
        with open(self.results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write(f'(VEP Extraction - Channels {self.channels})\n')
            f.write('-' * 30 + '\n')
            for i in range(self.channels):
                f.write(f'Channel {i + 1}: True VEPs = {true_veps_counts[i]}, False VEPs = {false_veps_counts[i]}\n')
            f.write('=' * 30 + '\n')
            f.write('Overall VEP Extraction Results\n')
            f.write('=' * 30 + '\n')
            f.write(f'Total True VEPs: {sum(true_veps_counts)}\n')
            f.write(f'Total False VEPs: {sum(false_veps_counts)}\n')
            f.write('=' * 30 + '\n\n')

    def remove_artifacts(self, data):
        for i in range(data.shape[0]):
            artifact_indices = np.where(np.abs(data[i]) > self.artifact_threshold)[0]
            for idx in artifact_indices:
                start = max(0, idx - self.artifact_window)
                end = min(len(data[i]), idx + self.artifact_window)
                data[i, start:end] = 0
        return data

    def extract_true_veps(self, data):
        true_veps = []
        for stim in self.stim_times:
            end_window = min(stim + self.local_peak_window, len(data))
            peak_idx = np.argmax(data[stim:end_window]) + stim  # Local peak index
            start_response = max(0, peak_idx - self.response_window)
            end_response = min(len(data), peak_idx + self.response_window)
            true_veps.append((start_response, end_response))
        return true_veps

    def detect_vep_responses(self, data, threshold_high, threshold_low, search_window, response_window):
        detected_responses = []
        i = 0
        while i < len(data):
            if data[i] > threshold_high:
                end_window = min(i + search_window, len(data))
                peak_idx = np.argmax(data[i:end_window]) + i
                start_check = max(0, peak_idx - 50)
                end_check = peak_idx
                if np.any(data[start_check:end_check] < threshold_low):
                    start_response = max(0, peak_idx - response_window)
                    end_response = min(len(data), peak_idx + response_window)
                    detected_responses.append((start_response, end_response))
                    i = end_response
                else:
                    i += 1
            else:
                i += 1
        return detected_responses

    def check_true_veps_and_collect_periods(self, detected_responses, true_veps, data):
        false_periods = []
        true_periods = []

        # Create a mask to mark true VEP segments in the data
        vep_mask = np.zeros(len(data), dtype=bool)
        for start, end in true_veps:
            vep_mask[start:end] = True
            true_periods.append(data[start:end].tolist())

        # Identify false VEPs in non-true VEP regions
        for start, end in detected_responses:
            if not vep_mask[start:end].any():
                false_periods.append(data[start:end].tolist())

        return {
            "true_time_periods": true_periods,
            "false_time_periods": false_periods
        }, len(true_periods), len(false_periods)

    def process_data(self):
        # Create directory for event dictionaries if it doesn't exist
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)

        # Preprocess the EEG data
        vep_mat_cleaned = self.remove_artifacts(self.vep_csv.values.T)

        total_true_veps = 0
        total_false_veps = 0
        true_veps_counts = []
        false_veps_counts = []

        for i in range(min(self.channels, vep_mat_cleaned.shape[0])):  # Iterate over each channel up to the specified limit
            # Extract True VEPs
            true_veps = self.extract_true_veps(vep_mat_cleaned[i])

            # Detect False VEP responses in remaining data
            detected_responses = self.detect_vep_responses(vep_mat_cleaned[i], 30, -13, 100, 100)

            # Check and collect true and false VEP periods
            response_dict, true_veps_count, false_veps_count = self.check_true_veps_and_collect_periods(detected_responses, true_veps, vep_mat_cleaned[i])

            # Update total counts and individual channel counts
            total_true_veps += true_veps_count
            total_false_veps += false_veps_count
            true_veps_counts.append(true_veps_count)
            false_veps_counts.append(false_veps_count)

            # Save the response dictionary to a JSON file
            with open(os.path.join(self.output_location, f'response_dict_channel_{i}.json'), 'w') as f:
                json.dump(response_dict, f)

        # Log results for all channels
        self.log_results(true_veps_counts, false_veps_counts)
