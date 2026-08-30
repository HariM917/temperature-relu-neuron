# Temperature Monitoring Neural Network Using ReLU

## Problem Statement

A neural network for temperature monitoring frequently receives negative pre-activation values. Implement a simple neuron using ReLU and test it with positive and negative temperature-related input values. Analyze the resulting activation outputs, compare the responses for positive and negative inputs, and determine how ReLU affects the representation of the input values.

## Objective

- Implement a single artificial neuron manually using Python
- Apply the ReLU (Rectified Linear Unit) activation function to temperature data
- Analyze how ReLU handles positive and negative pre-activation values
- Visualize the effect of ReLU on neuron outputs
- Understand why ReLU introduces non-linearity and sparse representations

## Concept Used

### Artificial Neuron

An artificial neuron is the basic building block of a neural network. It takes one or more inputs, multiplies each by a weight, adds a bias, and passes the result through an activation function.

### Weight (w)

The weight determines how much influence the input has on the output. A higher weight means the input has a stronger effect on the neuron's pre-activation value.

### Bias (b)

The bias is an additional parameter that shifts the pre-activation value. It allows the neuron to activate even when the input is zero, providing flexibility in learning.

### Pre-activation (z)

The pre-activation is the raw output of the neuron before applying the activation function:

```
z = w * x + b
```

Where:
- `x` = input value (temperature)
- `w` = weight
- `b` = bias
- `z` = pre-activation value

### ReLU (Rectified Linear Unit)

ReLU is one of the most widely used activation functions in neural networks:

```
ReLU(z) = max(0, z)
```

- If `z < 0`: ReLU(z) = 0 (negative values are clipped to zero)
- If `z >= 0`: ReLU(z) = z (positive values pass through unchanged)

### Why ReLU is Useful

1. **Simplicity**: ReLU is computationally efficient — just a comparison with zero
2. **Non-linearity**: Despite its simple formula, ReLU introduces non-linearity into the network
3. **Sparsity**: By outputting zero for negative inputs, ReLU creates sparse activations, meaning only a subset of neurons are active at any time
4. **Avoids vanishing gradient**: Unlike sigmoid or tanh, ReLU does not saturate for positive values

## Methodology

```
Temperature Input (x)
        ↓
Weighted Sum (w * x)
        ↓
Add Bias (w * x + b)
        ↓
Pre-activation (z)
        ↓
ReLU Activation: max(0, z)
        ↓
Activation Output
```

**Step-by-step process:**

1. Load temperature values from the CSV dataset
2. Define neuron parameters: weight (`w = 0.5`) and bias (`b = -8`)
3. Calculate the pre-activation: `z = 0.5 * temperature + (-8)`
4. Apply ReLU: `output = max(0, z)`
5. Display results in a comparison table
6. Visualize pre-activation vs. ReLU output

## Dataset

The dataset contains 8 temperature values in Celsius, including both negative and positive temperatures:

| Temperature (°C) |
|-------------------|
| -20               |
| -10               |
| 0                 |
| 5                 |
| 10                |
| 20                |
| 30                |
| 40                |

These values were chosen to demonstrate how the neuron responds to a range of real-world temperatures.

## Implementation

### Neuron Calculation

```python
weight = 0.5
bias = -8
pre_activation = weight * temperature + bias
```

### ReLU Function

```python
def relu(z):
    return np.maximum(0, z)
```

### Example Calculations

**For Temperature = -20°C:**
```
z = (0.5 × -20) + (-8) = -10 + (-8) = -18
ReLU(-18) = max(0, -18) = 0
```

**For Temperature = 40°C:**
```
z = (0.5 × 40) + (-8) = 20 + (-8) = 12
ReLU(12) = max(0, 12) = 12
```

## Results

| Temperature (°C) | Weighted Input | Bias | Pre-activation (z) | ReLU Output |
|-------------------|---------------|------|---------------------|-------------|
| -20               | -10.0         | -8   | -18.0               | 0.0         |
| -10               | -5.0          | -8   | -13.0               | 0.0         |
| 0                 | 0.0           | -8   | -8.0                | 0.0         |
| 5                 | 2.5           | -8   | -5.5                | 0.0         |
| 10                | 5.0           | -8   | -3.0                | 0.0         |
| 20                | 10.0          | -8   | 2.0                 | 2.0         |
| 30                | 15.0          | -8   | 7.0                 | 7.0         |
| 40                | 20.0          | -8   | 12.0                | 12.0        |

**Key Observation:** Temperatures below approximately 16°C produce negative pre-activation values, which ReLU clips to zero. Only temperatures above this threshold produce non-zero activation outputs.

## Analysis

### Response to Negative Pre-activation
When the pre-activation value `z` is negative (temperatures -20°C to 10°C in our example), ReLU outputs exactly zero. This means the neuron does not fire for these inputs.

### Response to Positive Pre-activation
When `z` is positive (temperatures 20°C, 30°C, 40°C), ReLU passes the value through unchanged. The neuron is "active" for these inputs.

### Effect of ReLU
ReLU acts as a gate: it blocks all negative pre-activation values and allows positive ones to pass. This creates a clear threshold behavior in the neuron.

### Sparse Representation
In our example, 5 out of 8 inputs produce zero output. This demonstrates **sparsity** — only a fraction of neurons in a network would be active for any given input. Sparse representations are computationally efficient and help the network learn meaningful features.

### Why ReLU Introduces Non-linearity
A linear function would produce outputs proportional to inputs across the entire range. ReLU breaks this linearity by clipping negative values to zero, creating a "bend" in the output curve. This non-linearity is essential because:
- Without it, stacking multiple neuron layers would be equivalent to a single linear transformation
- Non-linearity allows neural networks to learn complex, non-linear patterns in data

## Conclusion

This project demonstrates how a single artificial neuron with the ReLU activation function processes temperature data. The key findings are:

1. ReLU effectively filters out negative pre-activation values by clipping them to zero
2. Positive pre-activation values pass through unchanged, preserving their magnitude
3. The weight and bias parameters control which inputs activate the neuron
4. ReLU introduces non-linearity and sparsity, both essential properties for neural networks
5. The simplicity of ReLU (just `max(0, z)`) makes it both computationally efficient and easy to understand

## Technologies Used

| Technology   | Purpose                          |
|-------------|----------------------------------|
| Python       | Programming language             |
| NumPy        | Numerical calculations           |
| Pandas       | Data loading and manipulation    |
| Matplotlib   | Data visualization               |
| Jupyter Notebook | Interactive analysis         |

## Project Structure

```
temperature-relu-neuron/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── dataset/
│   └── temperature_data.csv   # Temperature input data
│
├── src/
│   └── relu_temperature_neuron.py   # Main Python script
│
├── notebooks/
│   └── ReLU_Temperature_Analysis.ipynb   # Jupyter notebook
│
├── results/
│   ├── relu_activation_plot.png              # ReLU visualization
│   └── temperature_activation_comparison.png  # Comparison chart
│
└── screenshots/
    └── output.png             # Terminal output screenshot
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Python Script

```bash
python src/relu_temperature_neuron.py
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook notebooks/ReLU_Temperature_Analysis.ipynb
```


## References

1. Nair, V., & Hinton, G. E. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6: Deep Feedforward Networks.
3. Glorot, X., Bordes, A., & Bengio, Y. (2011). "Deep Sparse Rectifier Neural Networks." *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
4. NumPy Documentation: https://numpy.org/doc/
5. Matplotlib Documentation: https://matplotlib.org/stable/contents.html
6. Pandas Documentation: https://pandas.pydata.org/docs/
