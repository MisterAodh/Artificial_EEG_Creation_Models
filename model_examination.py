import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import json

# Define the VEPNet model inside the class
class VEPNet(nn.Module):
    def __init__(self, input_size):
        super(VEPNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

        self._to_linear = None
        self.convs(torch.randn(1, 1, input_size))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def convs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Define the VEPModelExaminer class
class VEPModelExaminer:
    def __init__(self, augmented_data_dir="output_datasets/augmented_training_data", models_dir='models', results_path='output_datasets/results.txt'):
        self.augmented_data_dir = augmented_data_dir
        self.models_dir = models_dir
        self.results_path = results_path

    def load_augmented_data(self):
        """Load the augmented data and labels."""
        all_data = np.load(os.path.join(self.augmented_data_dir, 'all_data.npy'))
        all_labels = np.load(os.path.join(self.augmented_data_dir, 'all_labels.npy'))
        return all_data, all_labels

    def load_noise_signals(self):
        """Load the noise signals from noise_signals.json."""
        noise_file_path = os.path.join(self.augmented_data_dir, 'noise_signals.json')
        with open(noise_file_path, 'r') as f:
            noise_data = json.load(f)
        noise_signals = np.array(noise_data['noise_time_periods'])  # Correct key here
        noise_labels = np.zeros(len(noise_signals))  # Label all noise signals as 'fake' (0)
        return noise_signals, noise_labels

    def test_model(self, model, loader):
        """Evaluate the model using the provided DataLoader."""
        model.eval()
        correct_total = 0
        correct_original = 0
        total_total = 0
        total_original = 0
        false_positives_total = 0
        false_negatives_total = 0
        true_positives_total = 0
        true_negatives_total = 0
        false_positives_original = 0
        false_negatives_original = 0
        true_positives_original = 0
        true_negatives_original = 0

        with torch.no_grad():
            for inputs, labels in loader:
                flags = inputs[:, -1]  # Extract the original/augmented flag
                inputs = inputs[:, :-1].unsqueeze(1)  # Exclude the flag and add channel dimension
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()

                total_total += labels.size(0)
                correct_total += (predicted == labels).sum().item()

                original_mask = flags == 1
                if original_mask.sum() > 0:
                    total_original += original_mask.sum().item()
                    correct_original += (predicted[original_mask] == labels[original_mask]).sum().item()

                for i in range(len(predicted)):
                    # For total dataset
                    if predicted[i] == 1 and labels[i] == 0:
                        false_positives_total += 1
                    elif predicted[i] == 0 and labels[i] == 1:
                        false_negatives_total += 1
                    elif predicted[i] == 1 and labels[i] == 1:
                        true_positives_total += 1
                    elif predicted[i] == 0 and labels[i] == 0:
                        true_negatives_total += 1

                    # For unaugmented (original) dataset
                    if flags[i] == 1:
                        if predicted[i] == 1 and labels[i] == 0:
                            false_positives_original += 1
                        elif predicted[i] == 0 and labels[i] == 1:
                            false_negatives_original += 1
                        elif predicted[i] == 1 and labels[i] == 1:
                            true_positives_original += 1
                        elif predicted[i] == 0 and labels[i] == 0:
                            true_negatives_original += 1

        accuracy_total = 100 * correct_total / total_total if total_total > 0 else 0
        accuracy_original = 100 * correct_original / total_original if total_original > 0 else 0

        # Calculate additional metrics
        positive_accuracy_total = 100 * true_positives_total / (true_positives_total + false_positives_total) if (
                                                                                                                             true_positives_total + false_positives_total) > 0 else 0
        negative_accuracy_total = 100 * true_negatives_total / (true_negatives_total + false_negatives_total) if (
                                                                                                                             true_negatives_total + false_negatives_total) > 0 else 0
        positive_accuracy_original = 100 * true_positives_original / (
                    true_positives_original + false_positives_original) if (
                                                                                       true_positives_original + false_positives_original) > 0 else 0
        negative_accuracy_original = 100 * true_negatives_original / (
                    true_negatives_original + false_negatives_original) if (
                                                                                       true_negatives_original + false_negatives_original) > 0 else 0

        return {
            "total": {
                "accuracy": accuracy_total,
                "false_positives": false_positives_total,
                "false_negatives": false_negatives_total,
                "true_positives": true_positives_total,
                "true_negatives": true_negatives_total,
                "positive_accuracy": positive_accuracy_total,
                "negative_accuracy": negative_accuracy_total,
            },
            "original": {
                "accuracy": accuracy_original,
                "false_positives": false_positives_original,
                "false_negatives": false_negatives_original,
                "true_positives": true_positives_original,
                "true_negatives": true_negatives_original,
                "positive_accuracy": positive_accuracy_original,
                "negative_accuracy": negative_accuracy_original,
            }
        }

    def evaluate_model(self, model_number, noise, model_type):
        # Load the augmented data

        model_directory = self.models_dir
        if noise =='n':
            model_directory = model_directory + '/CNN_no_noise'
        else:
            model_directory = model_directory + '/' + model_type + '_noise'

        all_data, all_labels = self.load_augmented_data()

        # Combine the data and flags
        original_flags = np.array([1] * (len(all_labels) // 2) + [0] * (len(all_labels) // 2))
        full_dataset = torch.utils.data.TensorDataset(torch.tensor(np.column_stack((all_data, original_flags)), dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32))
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

        # Load the noise signals
        noise_data, noise_labels = self.load_noise_signals()
        noise_flags = np.array([0] * len(noise_labels))  # Label all noise signals as 'augmented'
        noise_dataset = torch.utils.data.TensorDataset(torch.tensor(np.column_stack((noise_data, noise_flags)), dtype=torch.float32), torch.tensor(noise_labels, dtype=torch.float32))
        noise_loader = DataLoader(noise_dataset, batch_size=32, shuffle=False)

        # Load the model
        model_name = f'model_{model_number}.pth'
        model_path = os.path.join(model_directory, model_name)
        if not os.path.exists(model_path):
            print(f"Model {model_number} does not exist.")
            return

        model = self.load_model(model_path, all_data.shape[1])

        # Test the model on the augmented data
        results_augmented = self.test_model(model, full_loader)
        self.print_and_log_results(model_number, model_path, results_augmented)

        # Test the model on the random noise signals
        results_noise = self.test_model(model, noise_loader)
        self.print_and_log_noise_results(model_number, model_path, results_noise)

    def load_model(self, model_path, input_size):
        """Load the model."""
        model = VEPNet(input_size=input_size)
        model.load_state_dict(torch.load(model_path))
        return model

    def print_and_log_results(self, model_number, model_path, results):
        """Log the evaluation results for augmented data to the results.txt file."""
        with open(self.results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write(f"Model {model_number} Evaluation on Augmented Data\n")
            f.write(f"Model Path: {model_path}\n")
            f.write("Total Dataset Results:\n")
            f.write(f"  Accuracy: {results['total']['accuracy']}%\n")
            f.write(f"  False Positives: {results['total']['false_positives']}\n")
            f.write(f"  False Negatives: {results['total']['false_negatives']}\n")
            f.write(f"  True Positives: {results['total']['true_positives']}\n")
            f.write(f"  True Negatives: {results['total']['true_negatives']}\n")
            f.write(f"  Positive Decision Accuracy: {results['total']['positive_accuracy']}%\n")
            f.write(f"  Negative Decision Accuracy: {results['total']['negative_accuracy']}%\n")
            f.write("Unaugmented Dataset Results:\n")
            f.write(f"  Accuracy: {results['original']['accuracy']}%\n")
            f.write(f"  False Positives: {results['original']['false_positives']}\n")
            f.write(f"  False Negatives: {results['original']['false_negatives']}\n")
            f.write(f"  True Positives: {results['original']['true_positives']}\n")
            f.write(f"  True Negatives: {results['original']['true_negatives']}\n")
            f.write(f"  Positive Decision Accuracy: {results['original']['positive_accuracy']}%\n")
            f.write(f"  Negative Decision Accuracy: {results['original']['negative_accuracy']}%\n")
            f.write('-' * 30 + '\n\n')

    def print_and_log_noise_results(self, model_number, model_path, results):
        """Log the evaluation results for noise signals to the results.txt file."""
        with open(self.results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write('-' * 30 + '\n')
            f.write(f"Model {model_number} Evaluation on Random Noise Data\n")
            f.write(f"Model Path: {model_path}\n")
            f.write("Noise Dataset Results:\n")
            f.write(f"  Accuracy: {results['total']['accuracy']}%\n")
            f.write(f"  False Positives: {results['total']['false_positives']}\n")
            f.write(f"  True Negatives: {results['total']['true_negatives']}\n")
            f.write('-' * 30 + '\n')
            f.write('-' * 30 + '\n\n')


    def run(self):
        noise = 'n'
        model_type = input("want to test CNN, RNN, CNN_RNN? :")
        if model_type == 'CNN':
            noise = input("want to test noise trained model? y/n")
        model_number = input("Enter the model number you want to evaluate: ")
        self.evaluate_model(model_number, noise, model_type)
