from Data_cleaning_functions import VEPDataProcessor, generate_noise_signals
from VEP_signal_extraction import VEPtrainingdata_extract
from model_creation_functions import create_identification_model, create_generator_model
from model_examination import VEPModelExaminer
import os

# Paths to the data files
vep_file_path = 'Initial_datasets/vep_data.csv'
stim_times_file_path = 'Initial_datasets/vep_stim_times.csv'
output_dir = 'output_datasets'
training_data_dir = os.path.join(output_dir, 'training_data')
augmented_data_dir = os.path.join(output_dir, 'augmented_training_data')

# Example usage
# processor = VEPDataProcessor(vep_file_path, stim_times_file_path)
# processor.process_and_display_data(sensor_index=2, output_dir=output_dir)
# processor.vep_analysis(output_dir=os.path.join(output_dir, 'average_channel_vep/'))

# Extract training data using the VEPtrainingdata_extract class
vep_data_location = os.path.join(output_dir, 'trimmed_vep_data.csv')
vep_stim_times_location = os.path.join(output_dir, 'trimmed_vep_stim_times.csv')
output_location = os.path.join(output_dir, 'training_data')

# Initialize the VEPtrainingdata_extract class and call the process_data method
# training_data_extractor = VEPtrainingdata_extract(vep_data_location, vep_stim_times_location, output_location, channels=10)
# training_data_extractor.process_data()
json_files = [f'output_datasets/training_data/{file}' for file in os.listdir('output_datasets/training_data/') if file.endswith('.json')]
create_generator_model(json_files, num_samples=1000, num_epochs=2, lr=0.0001)
# Ask the user if they want to generate a new model
# generate_new_model = input('Would you like to generate a new model? y / n: ')
#
# if generate_new_model.lower() == 'y':
#     use_noise = input('Would you like to include noise signals in the training? y / n: ')
#     use_noise_signals = True if use_noise.lower() == 'y' else False
#
#     # Load the event dictionaries JSON files created by VEPtrainingdata_extract
#     json_files = [f'output_datasets/training_data/{file}' for file in os.listdir('output_datasets/training_data/') if file.endswith('.json')]
#
#     model_type = input('What type of model would you like to generate?(CNN, RNN, CNN_RNN): ')
#     # Create the identification model with or without noise signals
#     create_identification_model(json_files, model_type=model_type)
#
# # Run the model examiner
# examiner = VEPModelExaminer()
# examiner.run()
