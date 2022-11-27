"""
Simple script to execute data generation for different systems & numbers of samples.
"""
from learning_gravity.util.data_generator import DataGenerator

FOLDER_NAME = "example_2"

if __name__ == "__main__":
    GENERATOR = DataGenerator(FOLDER_NAME)
    GENERATOR.generate_random_samples(num_samples=10_000, min_dist=.5)
