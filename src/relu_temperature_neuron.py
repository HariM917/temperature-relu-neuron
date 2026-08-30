import numpy as np
import pandas as pd
import os


# -------------------------------------------------------
# Step 1: Load the temperature dataset
# -------------------------------------------------------
def load_temperature_data(filepath):
    """Load temperature values from a CSV file."""
    data = pd.read_csv(filepath)
    return data


# -------------------------------------------------------
# Step 2: Define the neuron parameters
# -------------------------------------------------------
# Weight controls the influence of temperature.
# Bias shifts the pre-activation value.
weight = 0.5
bias = -8


# -------------------------------------------------------
# Step 3: Calculate the pre-activation value
# -------------------------------------------------------
def calculate_pre_activation(temperatures, weight, bias):
    """
    Calculate the pre-activation value for each temperature.

    Formula:
        z = w * x + b

    where:
        x = temperature
        w = weight
        b = bias
        z = pre-activation
    """
    weighted_input = weight * temperatures
    pre_activation = weighted_input + bias

    return weighted_input, pre_activation


# -------------------------------------------------------
# Step 4: Implement the ReLU activation function
# -------------------------------------------------------
def relu(z):
    """
    ReLU (Rectified Linear Unit).

    Formula:
        ReLU(z) = max(0, z)

    Negative values become 0.
    Positive values remain unchanged.
    """
    return np.maximum(0, z)


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
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

    # ---------------------------------------------------
    # Load temperature data
    # ---------------------------------------------------
    data = load_temperature_data(dataset_path)

    temperatures = data["temperature_celsius"].values

    print(
        f"Loaded {len(temperatures)} temperature values: "
        f"{temperatures}"
    )

    # ---------------------------------------------------
    # Calculate pre-activation
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Apply ReLU
    # ---------------------------------------------------
    activation_output = relu(pre_activation)

    print("\nReLU output:")
    print(activation_output)


# Run the program
if __name__ == "__main__":
    main()