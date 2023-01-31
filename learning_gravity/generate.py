"""
Simple script to execute data generation for different systems & numbers of samples.
"""
import argparse
from learning_gravity.util.data_generator import DataGenerator

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="gravity_learner")
    PARSER.add_argument(
        "--dataset",
        help="name of the dataset to generate data for",
        default="example_0"
    )
    PARSER.add_argument("--num-samples", type=int, default=10_000)
    PARSER.add_argument("--min-dist", type=float, default=0.5)
    ARGS = PARSER.parse_args()

    GENERATOR = DataGenerator(ARGS.dataset)
    GENERATOR.generate_random_samples(num_samples=ARGS.num_samples, min_dist=ARGS.min_dist)
