import numpy as np
import pandas as pd
import os


# Step 1: Load the temperature dataset
def load_temperature_data(filepath):
    """Load temperature values from a CSV file."""
    data = pd.read_csv(filepath)
    return data


# Step 2: Define neuron parameters
weight = 0.5
bias = -8


# Step 3: Calculate pre-activation
def calculate_pre_activation(temperatures, weight, bias):
    """
    Calculate the pre-activation value.

    Formula:
        z = w * x + b
    """
    weighted_input = weight * temperatures
    pre_activation = weighted_input + bias

    return weighted_input, pre_activation


def main():
    # Determine the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Path to temperature dataset
    dataset_path = os.path.join(
        project_dir,
        "dataset",
        "temperature_data.csv"
    )

    # Load dataset
    data = load_temperature_data(dataset_path)

    temperatures = data["temperature_celsius"].values

    print(
        f"Loaded {len(temperatures)} temperature values: "
        f"{temperatures}"
    )

    # Calculate neuron pre-activation
    weighted_input, pre_activation = calculate_pre_activation(
        temperatures,
        weight,
        bias
    )

    print("\nNeuron Parameters:")
    print(f"Weight = {weight}")
    print(f"Bias = {bias}")

    print("\nPre-activation values:")
    print(pre_activation)


if __name__ == "__main__":
    main()