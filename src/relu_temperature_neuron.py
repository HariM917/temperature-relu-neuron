import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
weight = 0.5
bias = -8


# -------------------------------------------------------
# Step 3: Calculate the pre-activation value
# -------------------------------------------------------
def calculate_pre_activation(temperatures, weight, bias):
    """
    Calculate the pre-activation value.

    Formula:
        z = w * x + b
    """
    weighted_input = weight * temperatures
    pre_activation = weighted_input + bias

    return weighted_input, pre_activation


# -------------------------------------------------------
# Step 4: Implement the ReLU activation function
# -------------------------------------------------------
def relu(z):
    """
    ReLU activation function.

    Formula:
        ReLU(z) = max(0, z)
    """
    return np.maximum(0, z)


# -------------------------------------------------------
# Step 5: Display results and analysis
# -------------------------------------------------------
def display_results(results_df):
    """Display neuron calculations and activation analysis."""

    print("\n" + "=" * 75)
    print("TEMPERATURE NEURON ACTIVATION RESULTS")
    print("=" * 75)

    print(f"\nNeuron Parameters:")
    print(f"  Weight = {weight}")
    print(f"  Bias   = {bias}")

    print("\nFormula:")
    print(f"  z = {weight} * temperature + ({bias})")

    print("\nActivation:")
    print("  ReLU(z) = max(0, z)")

    print("\n" + "-" * 75)

    print(
        f"{'Temperature (°C)':>18} | "
        f"{'Weighted Input':>15} | "
        f"{'Bias':>6} | "
        f"{'Pre-activation':>15} | "
        f"{'ReLU Output':>12}"
    )

    print("-" * 75)

    for _, row in results_df.iterrows():
        print(
            f"{row['Temperature (°C)']:>18.1f} | "
            f"{row['Weighted Input']:>15.1f} | "
            f"{row['Bias']:>6.1f} | "
            f"{row['Pre-activation (z)']:>15.1f} | "
            f"{row['ReLU Output']:>12.1f}"
        )

    print("-" * 75)

    # Summary analysis
    total = len(results_df)

    zero_count = (
        results_df["ReLU Output"] == 0
    ).sum()

    active_count = total - zero_count

    sparsity = zero_count / total

    print("\nSummary:")
    print(f"  Total inputs:      {total}")
    print(
        f"  Zero outputs:      {zero_count} "
        f"(neuron did NOT fire)"
    )
    print(
        f"  Non-zero outputs:  {active_count} "
        f"(neuron fired)"
    )
    print(
        f"  Sparsity:          "
        f"{zero_count}/{total} = {sparsity:.1%}"
    )

    print("=" * 75)


# -------------------------------------------------------
# Step 6: Create ReLU activation visualization
# -------------------------------------------------------
def plot_relu_activation(results_df, save_dir):
    """
    Plot temperature against pre-activation
    and ReLU output.
    """

    temperatures = results_df["Temperature (°C)"]
    pre_activation = results_df["Pre-activation (z)"]
    relu_output = results_df["ReLU Output"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        temperatures,
        pre_activation,
        "bo-",
        label="Pre-activation (z)",
        linewidth=2,
        markersize=8
    )

    plt.plot(
        temperatures,
        relu_output,
        "rs-",
        label="ReLU Output",
        linewidth=2,
        markersize=8
    )

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        linewidth=0.8
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Value")

    plt.title(
        "Temperature vs Pre-activation and ReLU Output"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(
        save_dir,
        "relu_activation_plot.png"
    )

    plt.savefig(filepath, dpi=150)

    print(f"\nPlot saved: {filepath}")

    plt.close()


# -------------------------------------------------------
# Step 7: Create comparison bar chart
# -------------------------------------------------------
def plot_activation_comparison(results_df, save_dir):
    """
    Create a bar chart comparing temperature,
    pre-activation and ReLU output.
    """

    temperatures = results_df["Temperature (°C)"]
    pre_activation = results_df["Pre-activation (z)"]
    relu_output = results_df["ReLU Output"]

    x = np.arange(len(temperatures))
    bar_width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - bar_width,
        temperatures,
        bar_width,
        label="Temperature (°C)",
        color="steelblue"
    )

    plt.bar(
        x,
        pre_activation,
        bar_width,
        label="Pre-activation (z)",
        color="orange"
    )

    plt.bar(
        x + bar_width,
        relu_output,
        bar_width,
        label="ReLU Output",
        color="green"
    )

    plt.xlabel("Input Temperature")
    plt.ylabel("Value")

    plt.title(
        "Comparison: Temperature, Pre-activation, and ReLU Output"
    )

    plt.xticks(
        x,
        [f"{t:.0f}°C" for t in temperatures]
    )

    plt.legend()

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        linewidth=0.8
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    filepath = os.path.join(
        save_dir,
        "temperature_activation_comparison.png"
    )

    plt.savefig(filepath, dpi=150)

    print(f"Plot saved: {filepath}")

    plt.close()


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
def main():

    # Determine project directory
    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_dir = os.path.dirname(script_dir)

    # Dataset path
    dataset_path = os.path.join(
        project_dir,
        "dataset",
        "temperature_data.csv"
    )

    # Results directory
    results_dir = os.path.join(
        project_dir,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    # ---------------------------------------------------
    # Load data
    # ---------------------------------------------------
    data = load_temperature_data(
        dataset_path
    )

    temperatures = data[
        "temperature_celsius"
    ].values

    print(
        f"Loaded {len(temperatures)} temperature values: "
        f"{temperatures}"
    )

    # ---------------------------------------------------
    # Calculate neuron values
    # ---------------------------------------------------
    weighted_input, pre_activation = (
        calculate_pre_activation(
            temperatures,
            weight,
            bias
        )
    )

    # ---------------------------------------------------
    # Apply ReLU
    # ---------------------------------------------------
    activation_output = relu(
        pre_activation
    )

    # ---------------------------------------------------
    # Build results table
    # ---------------------------------------------------
    results_df = pd.DataFrame({

        "Temperature (°C)": temperatures,

        "Weighted Input": weighted_input,

        "Bias": [bias] * len(temperatures),

        "Pre-activation (z)": pre_activation,

        "ReLU Output": activation_output
    })

    # ---------------------------------------------------
    # Display results
    # ---------------------------------------------------
    display_results(
        results_df
    )

    # ---------------------------------------------------
    # Generate visualizations
    # ---------------------------------------------------
    plot_relu_activation(
        results_df,
        results_dir
    )

    plot_activation_comparison(
        results_df,
        results_dir
    )

    print(
        "\nDone! Check the 'results/' folder "
        "for saved plots."
    )


# -------------------------------------------------------
# Run program
# -------------------------------------------------------
if __name__ == "__main__":
    main()