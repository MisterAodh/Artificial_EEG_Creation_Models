import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
# Define the Diffusion Network for VEP Signal Generation
# Define the Diffusion Network for VEP Signal Generation
class VEPDiffusionModel(nn.Module):
    def __init__(self, input_size=200, timesteps=100, noise_schedule="linear"):
        super(VEPDiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.noise_schedule = self.get_noise_schedule(timesteps, noise_schedule)

        # Define the neural network architecture
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def get_noise_schedule(self, timesteps, noise_schedule):
        if noise_schedule == "linear":
            return torch.linspace(1e-6, 1e-2, timesteps)  # Reduce the upper bound of noise
        elif noise_schedule == "cosine":
            return torch.cos(torch.linspace(0, torch.pi / 2, timesteps)) ** 2
        else:
            raise ValueError("Unsupported noise schedule")

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_t = self.noise_schedule[t].to(x0.device)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    def reverse_diffusion(self, xt, t):
        # Ensure input has the correct number of channels for the CNN
        if xt.dim() == 2:  # If xt is missing the channel dimension
            xt = xt.unsqueeze(1)  # Add the channel dimension
        pred_noise = self.net(xt)
        alpha_t = self.noise_schedule[t].to(xt.device)
        return (xt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

    def forward(self, x, t):
        xt = self.forward_diffusion(x, t)
        return self.reverse_diffusion(xt, t)

    def train_model(self, train_loader, num_epochs=100, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, _ in train_loader:
                inputs = inputs.unsqueeze(1)  # Ensure inputs have a channel dimension
                optimizer.zero_grad()
                loss = 0
                for t in range(self.timesteps):
                    noise_inputs = self.forward_diffusion(inputs, t)
                    outputs = self(noise_inputs, t)
                    if torch.isnan(outputs).any():  # Check for NaN values
                        print("NaN detected in outputs!")
                        return
                    loss += criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    def generate_samples(self, num_samples, input_size=200):
        self.eval()
        with torch.no_grad():
            samples = torch.randn((num_samples, 1, input_size)).to(next(self.parameters()).device)
            for t in reversed(range(self.timesteps)):
                samples = self.reverse_diffusion(samples, t)
        return samples.squeeze().cpu().numpy()

    def save_model(self, model_type='diffusion'):
        """Save the trained model to disk."""
        model_dir = f'models/generators/{model_type}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        existing_models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
        model_idx = len(existing_models)
        model_path = os.path.join(model_dir, f'model_{model_idx}.pth')
        torch.save(self.state_dict(), model_path)
        self.log_model_saving(model_path)
        return model_path

    def log_model_saving(self, model_path):
        """Log the model saving event."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Generator Model saved to {model_path}\n")

# Define the function to create a diffusion generator model
def create_generator_model(json_files, num_samples=1000, num_epochs=100, lr=0.001):
    # Initialize the training data management class
    data_manager = VEPTrainingDataManagement(json_files)

    # Load the saved augmented data
    all_data = data_manager.all_data  # Load from class instance
    all_labels = data_manager.all_labels  # Load from class instance

    # Convert data to TensorDataset and DataLoader
    train_dataset = TensorDataset(torch.tensor(all_data, dtype=torch.float32),
                                  torch.tensor(all_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the diffusion model
    model = VEPDiffusionModel(input_size=all_data.shape[1])

    # Train the model
    model.train_model(train_loader, num_epochs=num_epochs, lr=lr)

    # Save the model
    model_path = model.save_model()

    # Generate samples
    generated_samples = model.generate_samples(num_samples, input_size=all_data.shape[1])

    # Save the generated samples to a JSON file
    output_dir = "output_datasets/generated_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "generated_signals.json")
    with open(output_file, 'w') as f:
        json.dump({"generated_signals": generated_samples.tolist()}, f)

    print(f"Generated signals saved to {output_file}")

    # Log the results
    with open(os.path.join('output_datasets', 'results.txt'), 'a') as f:
        f.write('-' * 30 + '\n')
        f.write(f"Generator Model Created\n")
        f.write(f"Model saved to: {model_path}\n")
        f.write(f"Generated signals saved to: {output_file}\n")
        f.write('-' * 30 + '\n\n')



# New CRNN Model for VEP Identification
class VEPCRNNModel(nn.Module):
    def __init__(self, input_size, conv_output_size=256, hidden_size=128, num_layers=2):
        super(VEPCRNNModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, conv_output_size, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        # RNN layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(conv_output_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 because of bidirectionality
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(0.5)

    def convs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        return x

    def forward(self, x):
        # Apply CNN layers
        x = self.convs(x)
        x = x.permute(0, 2, 1)  # Change dimension to (batch_size, sequence_length, input_size)

        # Apply RNN layers
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectionality
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.rnn(x, (h0, c0))

        # Apply fully connected layers
        x = self.dropout2(torch.relu(self.fc1(x[:, -1, :])))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

    def train_model(self, train_loader, noise_loader, num_epochs=20, lr=0.001):
        """Train the model with both regular and noise data loaders."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            # Train on regular data
            for inputs, labels in train_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension for CNN
                inputs = inputs.view(inputs.size(0), 1, -1)  # Reshape to (batch_size, 1, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Train on noise data
            for inputs, labels in noise_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension for CNN
                inputs = inputs.view(inputs.size(0), 1, -1)  # Reshape to (batch_size, 1, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.log_training_epoch(epoch + 1, running_loss / (len(train_loader) + len(noise_loader)))

    def log_training_epoch(self, epoch, loss):
        """Log the training loss for each epoch."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {loss}\n")

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        false_positives = []
        false_negatives = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension for CNN
                inputs = inputs.view(inputs.size(0), 1, -1)  # Reshape to (batch_size, 1, input_size)
                outputs = self(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect false positives and false negatives
                for i in range(len(predicted)):
                    if predicted[i] == 1 and labels[i] == 0:
                        false_positives.append(inputs[i].squeeze().tolist())
                    elif predicted[i] == 0 and labels[i] == 1:
                        false_negatives.append(inputs[i].squeeze().tolist())

        accuracy = 100 * correct / total
        self.log_evaluation(accuracy, len(false_positives), len(false_negatives))
        return accuracy, false_positives, false_negatives

    def log_evaluation(self, accuracy, false_positives, false_negatives):
        """Log the evaluation results."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(
                f"Evaluation Results: Accuracy = {accuracy}%, False Positives = {false_positives}, False Negatives = {false_negatives}\n")

    def save_model(self, model_type='CRNN'):
        """Save the trained model to disk."""
        model_dir = f'models/{model_type}_noise'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        existing_models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
        model_idx = len(existing_models)
        model_path = os.path.join(model_dir, f'model_{model_idx}.pth')
        torch.save(self.state_dict(), model_path)
        self.log_model_saving(model_path)
        return model_path

    def log_model_saving(self, model_path):
        """Log the model saving event."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Model saved to {model_path}\n")




# Updated RNN Model for VEP Identification with noise
class VEPRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(VEPRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 because of bidirectionality
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectionality
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        x, _ = self.rnn(x, (h0, c0))
        x = self.dropout(torch.relu(self.fc1(x[:, -1, :])))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

    def train_model(self, train_loader, noise_loader, num_epochs=20, lr=0.001):
        """Train the model with both regular and noise data loaders."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            # Train on regular data
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1, inputs.size(1))  # Reshape to (batch_size, sequence_length, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Train on noise data
            for inputs, labels in noise_loader:
                inputs = inputs.view(inputs.size(0), -1, inputs.size(1))  # Reshape to (batch_size, sequence_length, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.log_training_epoch(epoch + 1, running_loss / (len(train_loader) + len(noise_loader)))

    def log_training_epoch(self, epoch, loss):
        """Log the training loss for each epoch."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {loss}\n")

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        false_positives = []
        false_negatives = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.size(0), -1, inputs.size(1))  # Reshape for RNN
                outputs = self(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect false positives and false negatives
                for i in range(len(predicted)):
                    if predicted[i] == 1 and labels[i] == 0:
                        false_positives.append(inputs[i].squeeze().tolist())
                    elif predicted[i] == 0 and labels[i] == 1:
                        false_negatives.append(inputs[i].squeeze().tolist())

        accuracy = 100 * correct / total
        self.log_evaluation(accuracy, len(false_positives), len(false_negatives))
        return accuracy, false_positives, false_negatives

    def log_evaluation(self, accuracy, false_positives, false_negatives):
        """Log the evaluation results."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(
                f"Evaluation Results: Accuracy = {accuracy}%, False Positives = {false_positives}, False Negatives = {false_negatives}\n")

    def save_model(self, model_type='RNN'):
        """Save the trained model to disk."""
        model_dir = f'models/{model_type}_noise'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        existing_models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
        model_idx = len(existing_models)
        model_path = os.path.join(model_dir, f'model_{model_idx}.pth')
        torch.save(self.state_dict(), model_path)
        self.log_model_saving(model_path)
        return model_path

    def log_model_saving(self, model_path):
        """Log the model saving event."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Model saved to {model_path}\n")

# Define the VEP Training Data Management class
class VEPTrainingDataManagement:
    def __init__(self, json_files, output_dir="output_datasets/augmented_training_data"):
        self.json_files = json_files
        self.output_dir = output_dir
        self.true_data = []
        self.false_data = []
        self.max_len = 0
        self.load_data()
        self.pad_sequences()
        self.augment_data()
        self.save_augmented_data()

    def load_data(self):
        """Load the response dictionary from JSON files."""
        for file in self.json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                self.true_data.extend(data['true_time_periods'])
                self.false_data.extend(data['false_time_periods'])
        self.max_len = max(max(len(seq) for seq in self.true_data), max(len(seq) for seq in self.false_data))

    def pad_sequences(self):
        """Pad the sequences to ensure uniform length."""
        self.true_data = [seq + [0] * (self.max_len - len(seq)) if len(seq) < self.max_len else seq[:self.max_len] for
                          seq in self.true_data]
        self.false_data = [seq + [0] * (self.max_len - len(seq)) if len(seq) < self.max_len else seq[:self.max_len] for
                           seq in self.false_data]

    def augment_data(self):
        """Augment the data by adding Gaussian noise."""
        self.augmented_true_data = [seq + np.random.normal(0, 5, len(seq)) for seq in self.true_data]
        self.augmented_false_data = [seq + np.random.normal(0, 5, len(seq)) for seq in self.false_data]

    def save_augmented_data(self):
        """Save the augmented data and log the results."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.all_data = np.concatenate(
            (self.true_data, self.false_data, self.augmented_true_data, self.augmented_false_data), axis=0)
        self.all_labels = np.concatenate((np.ones(len(self.true_data)), np.zeros(len(self.false_data)),
                                          np.ones(len(self.augmented_true_data)),
                                          np.zeros(len(self.augmented_false_data))), axis=0)

        np.save(os.path.join(self.output_dir, 'all_data.npy'), self.all_data)
        np.save(os.path.join(self.output_dir, 'all_labels.npy'), self.all_labels)

        self.log_augmentation(len(self.true_data), len(self.false_data), len(self.augmented_true_data),
                              len(self.augmented_false_data))

    def log_augmentation(self, true_count, false_count, augmented_true_count, augmented_false_count):
        """Log the augmentation results."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write('-' * 30 + '\n')
            f.write('Augmented Training Data Creation\n')
            f.write('-' * 30 + '\n')
            f.write(f'Real True VEPs: {true_count}, Real False VEPs: {false_count}\n')
            f.write(f'Augmented True VEPs: {augmented_true_count}, Augmented False VEPs: {augmented_false_count}\n')
            f.write('-' * 30 + '\n\n')

    def load_noise_data(self):
        """Load noise signals from the noise_signals.json file."""
        noise_file = os.path.join(self.output_dir, 'noise_signals.json')
        if not os.path.exists(noise_file):
            raise FileNotFoundError("Noise signals file not found.")

        with open(noise_file, 'r') as f:
            noise_data = json.load(f)['noise_time_periods']

        noise_labels = np.zeros(len(noise_data))  # All noise signals should be labeled as '0'

        # Pad the noise signals to match the max_len
        noise_data = [seq + [0] * (self.max_len - len(seq)) if len(seq) < self.max_len else seq[:self.max_len] for seq
                      in noise_data]
        noise_data = np.array(noise_data)

        return noise_data, noise_labels

    def combine_with_noise_data(self, noise_data, noise_labels):
        """Combine the existing data with noise data."""
        self.all_data = np.concatenate((self.all_data, noise_data), axis=0)
        self.all_labels = np.concatenate((self.all_labels, noise_labels), axis=0)


# Define the VEP Identification Model class
class VEPIdentificationModel(nn.Module):
    def __init__(self, input_size):
        super(VEPIdentificationModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

        # Compute the size of the flattened feature map after conv and pool layers
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

    def train_model(self, train_loader, num_epochs=20, lr=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.log_training_epoch(epoch + 1, running_loss / len(train_loader))

    def train_model_with_noise(self, train_loader, noise_loader, num_epochs=20, lr=0.001):
        """Train the model with both regular and noise data loaders."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1, 200)  # Reshape to (batch_size, sequence_length, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            for inputs, labels in noise_loader:
                inputs = inputs.view(inputs.size(0), -1, 200)  # Reshape to (batch_size, sequence_length, input_size)
                labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.log_training_epoch(epoch + 1, running_loss / (len(train_loader) + len(noise_loader)))


    def log_training_epoch(self, epoch, loss):
        """Log the training loss for each epoch."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {loss}\n")

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        false_positives = []
        false_negatives = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
                outputs = self(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect false positives and false negatives
                for i in range(len(predicted)):
                    if predicted[i] == 1 and labels[i] == 0:
                        false_positives.append(inputs[i].squeeze().tolist())
                    elif predicted[i] == 0 and labels[i] == 1:
                        false_negatives.append(inputs[i].squeeze().tolist())

        accuracy = 100 * correct / total
        self.log_evaluation(accuracy, len(false_positives), len(false_negatives))
        return accuracy, false_positives, false_negatives

    def log_evaluation(self, accuracy, false_positives, false_negatives):
        """Log the evaluation results."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(
                f"Evaluation Results: Accuracy = {accuracy}%, False Positives = {false_positives}, False Negatives = {false_negatives}\n")

    def save_model(self, noise, C_CR_R):
        """Save the trained model to disk."""
        if C_CR_R == 'CNN':
            if noise:
                model_dir = 'models/CNN_noise'
            else:
                model_dir = 'models/CNN_no_noise'
        if C_CR_R == 'RNN':
            model_dir = 'models/RNN_noise'
        if C_CR_R == 'CRNN':
            model_dir = 'models/CNN_RNN_noise'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        existing_models = [f for f in os.listdir(model_dir) if f.startswith('model_')]
        model_idx = len(existing_models)
        model_path = os.path.join(model_dir, f'model_{model_idx}.pth')
        torch.save(self.state_dict(), model_path)
        self.log_model_saving(model_path)
        return model_path

    def log_model_saving(self, model_path):
        """Log the model saving event."""
        results_path = os.path.join('output_datasets', 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Model saved to {model_path}\n")


# Define the function to create an identification model
# Modified create_identification_model function to handle RNN
# Function to create the CRNN-based identification model
def create_identification_model(json_files, num_epochs=80, lr=0.001, model_type='CRNN'):
    # Initialize the training data management class
    data_manager = VEPTrainingDataManagement(json_files)

    # Load the saved augmented data
    all_data = data_manager.all_data  # Load from class instance
    all_labels = data_manager.all_labels  # Load from class instance

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

    # Convert data to TensorDataset and DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the CRNN model
    model = VEPCRNNModel(input_size=all_data.shape[1])

    # Load noise data
    noise_data, noise_labels = data_manager.load_noise_data()
    noise_dataset = TensorDataset(torch.tensor(noise_data, dtype=torch.float32), torch.tensor(noise_labels, dtype=torch.float32))
    noise_loader = DataLoader(noise_dataset, batch_size=32, shuffle=True)

    # Train the model with noise
    model.train_model(train_loader, noise_loader, num_epochs=num_epochs, lr=lr)

    # Evaluate the model
    accuracy, false_positives, false_negatives = model.evaluate_model(test_loader)

    # Save the model
    model_path = model.save_model(model_type=model_type)

    # Log the results
    with open(os.path.join('output_datasets', 'results.txt'), 'a') as f:
        f.write('-' * 30 + '\n')
        f.write(f"Identification Model Created\n")
        f.write(f"Model saved to: {model_path}\n")
        f.write(f"Preliminary Evaluation Accuracy: {accuracy}%\n")
        f.write('-' * 30 + '\n\n')
