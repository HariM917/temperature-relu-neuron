import pandas as pd
import os


# Step 1: Load the temperature dataset
def load_temperature_data(filepath):
    """Load temperature values from a CSV file."""
    data = pd.read_csv(filepath)
    return data


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

    # Extract temperature values
    temperatures = data["temperature_celsius"].values

    print(
        f"Loaded {len(temperatures)} temperature values: "
        f"{temperatures}"
    )


if __name__ == "__main__":
    main()