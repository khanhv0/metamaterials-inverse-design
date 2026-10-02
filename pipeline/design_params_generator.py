import numpy as np
import pandas as pd
from scipy.stats import qmc

class LHSSampler:
    """
    A class for generating samples across given ranges of parameters using LHS for uniform coverage
    of each parameter range
    """
    def __init__(self, param_ranges):
        """
        param_ranges: dict
            Dictionary defining parameter ranges.
            Example:
                {
                    'w': ([0.8, 0.5, 0.5, 0.25], [3, 3, 3, 3]),
                    'h4': (7, 14),
                    'd1': (1, 3),
                    'd2': (0.2, 1),
                    'd3': (0.2, 1)
                }
        """
        self.param_ranges = param_ranges
        self.param_names = []
        self.low_bounds = []
        self.high_bounds = []
        self._prepare_bounds()

    def _prepare_bounds(self):
        """Flatten list parameters and store bounds."""
        for name, (low, high) in self.param_ranges.items():
            if isinstance(low, (list, tuple)):
                for i, (l, h) in enumerate(zip(low, high)):
                    self.param_names.append(f"{name}{i+1}") 
                    self.low_bounds.append(l)
                    self.high_bounds.append(h)
            else:
                self.param_names.append(name)
                self.low_bounds.append(low)
                self.high_bounds.append(high)
        self.low_bounds = np.array(self.low_bounds)
        self.high_bounds = np.array(self.high_bounds)

    def sample(self, n = 1000, csv_filename=None, random_state=None):
        """
        Generate n Latin Hypercube Samples.
        
        Parameters
        ----------
        n : int
            Number of samples to generate.
        csv_filename : str, optional
            Path to save the CSV file (e.g., "samples.csv").
        random_state : int, optional
            Seed for reproducibility.
        
        Returns
        -------
        samples : np.ndarray
            Array of shape (n, num_parameters).
        param_names : list
            List of parameter names in the same order as columns.
        """
        sampler = qmc.LatinHypercube(d=len(self.param_names), seed=random_state)
        lhs = sampler.random(n)
        scaled_samples = qmc.scale(lhs, self.low_bounds, self.high_bounds)

        if csv_filename:
            df = pd.DataFrame(scaled_samples, columns=self.param_names)
            df.to_csv(csv_filename, index=False)

        return scaled_samples, self.param_names
