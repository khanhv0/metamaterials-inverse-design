import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def verify_designs_in_range(predicted_designs):
    lower_bound = np.array([1, 0.5, 0.5, 0.25, 7, 0.3, 0.2, 0.2])
    upper_bound = np.array([2.8, 2.5, 3, 1.19, 10.5, 1, 0.85, 1.5])

    # row × parameter violations
    violations = (predicted_designs < lower_bound) | (predicted_designs > upper_bound)

    return violations

def main():
    df1 = pd.read_csv("pipeline/inverse_model_old_validation.csv")
    df2 = pd.read_csv("outputs/properties_test_samples_results.csv")

    cols1 = ['w_1', 'w_2', 'w_3', 'w_4', 'h_4', 'd_1', 'd_2', 'd_3']

    # Convert to numpy
    X1 = df1.iloc[:, :8].to_numpy()
    X2 = df2.iloc[:, :8].to_numpy()

    # Get row-level and parameter-level violations
    violations1 = verify_designs_in_range(X1)
    violations2 = verify_designs_in_range(X2)

    # --- Existing bar plot (dataset-level) ---
    count_true1 = np.any(violations1, axis=1).sum()
    count_true2 = np.any(violations2, axis=1).sum()

    print(count_true1, count_true2)

    dataset_names = ["System A", "System B"]
    amount_of_trues = [count_true1, count_true2]

    plt.figure()

    bars = plt.bar(dataset_names, amount_of_trues)

    for bar, val in zip(bars, amount_of_trues):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{val}",
            ha='center', va='bottom',
            fontsize=8
        )

    plt.title("Amount of designs outside the predetermined ranges")
    plt.show()

    # --- NEW: Parameter-level violation counts ---
    param_violation_counts1 = violations1.sum(axis=0)
    param_violation_counts2 = violations2.sum(axis=0)

    parameters = cols1

    # Bar plot comparing parameter violations
    plt.figure()
    bar_width = 0.35
    x = np.arange(len(parameters))

    plt.bar(x - bar_width/2, param_violation_counts1, width=bar_width, label="System A")
    plt.bar(x + bar_width/2, param_violation_counts2, width=bar_width, label="System B")

    plt.xticks(x, parameters)
    plt.ylabel("Number of violations")
    plt.title("Out of bounds distributions per design parameter")
    plt.legend()
    plt.tight_layout()
    plt.show()


main()