"""
Parameter grid system for systematic parameter exploration and optimization.
"""

import itertools
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class MethodConfig:
    """Configuration for a specific method."""

    method_class: Any  # The actual method function/class
    parameter_space: Dict[str, List[Any]] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_constraints: Dict[str, Any] = field(default_factory=dict)

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters against constraints.

        Returns:
            True if parameters are valid, raises ValueError if invalid
        """
        for param_name, param_value in parameters.items():
            if param_name in self.parameter_constraints:
                constraint = self.parameter_constraints[param_name]
                self._validate_single_constraint(param_name, param_value, constraint)
        return True

    def _validate_single_constraint(self, name: str, value: Any, constraint: Dict[str, Any]):
        """Validate a single parameter against constraints."""

        if 'type' in constraint:
            expected_type = constraint['type']
            if not isinstance(value, expected_type):
                raise ValueError(f"Parameter {name} must be of type {expected_type}, got {type(value)}")

        if 'range' in constraint:
            min_val, max_val = constraint['range']
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {name}={value} out of range [{min_val}, {max_val}]")

        if 'choices' in constraint:
            if value not in constraint['choices']:
                raise ValueError(f"Parameter {name}={value} not in choices {constraint['choices']}")

        if 'min' in constraint and value < constraint['min']:
            raise ValueError(f"Parameter {name}={value} below minimum {constraint['min']}")

        if 'max' in constraint and value > constraint['max']:
            raise ValueError(f"Parameter {name}={value} above maximum {constraint['max']}")


class ParameterGrid:
    """
    Manages parameter combinations for systematic exploration.

    Supports multiple generation strategies:
    - Grid search: Exhaustive combination of all parameter values
    - Random search: Random sampling of parameter combinations
    - Bayesian optimization: Smart parameter selection (simplified)
    """

    def __init__(
        self,
        param_dict: Dict[str, Union[List[Any], Tuple[float, float]]],
        continuous_params: Optional[List[str]] = None
    ):
        """
        Initialize parameter grid.

        Args:
            param_dict: Dictionary of parameter_name -> parameter_values or (min, max) range
            continuous_params: List of parameters that are continuous (for range specifications)
        """
        self.param_dict = param_dict
        self.continuous_params = continuous_params or []
        self.combinations = []

    def generate_combinations(
        self,
        strategy: str = 'grid',
        n_samples: int = 100,
        n_bins: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations.

        Args:
            strategy: 'grid', 'random', or 'bayesian'
            n_samples: Number of samples for random/bayesian strategies
            n_bins: Number of bins for discretizing continuous ranges in grid search

        Returns:
            List of parameter dictionaries
        """
        if strategy == 'grid':
            return self._generate_grid_combinations(n_bins)
        elif strategy == 'random':
            return self._generate_random_combinations(n_samples)
        elif strategy == 'bayesian':
            return self._generate_bayesian_combinations(n_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_grid_combinations(self, n_bins: int = 10) -> List[Dict[str, Any]]:
        """Generate exhaustive grid search combinations."""

        # Process parameter dictionary
        param_lists = {}

        for param_name, param_values in self.param_dict.items():
            if isinstance(param_values, tuple) and len(param_values) == 2:
                # Continuous range
                min_val, max_val = param_values
                if param_name in self.continuous_params:
                    param_lists[param_name] = np.linspace(min_val, max_val, n_bins).tolist()
                else:
                    # Discrete range
                    param_lists[param_name] = list(range(int(min_val), int(max_val) + 1))
            else:
                # List of discrete values
                param_lists[param_name] = param_values

        # Generate all combinations
        keys = list(param_lists.keys())
        values = list(param_lists.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        self.combinations = combinations
        return combinations

    def _generate_random_combinations(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""

        combinations = []

        for _ in range(n_samples):
            param_dict = {}

            for param_name, param_values in self.param_dict.items():
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous range
                    min_val, max_val = param_values
                    if param_name in self.continuous_params:
                        param_dict[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        # Discrete range
                        param_dict[param_name] = np.random.randint(int(min_val), int(max_val) + 1)
                else:
                    # Random choice from list
                    param_dict[param_name] = np.random.choice(param_values)

            combinations.append(param_dict)

        self.combinations = combinations
        return combinations

    def _generate_bayesian_combinations(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate combinations using simplified Bayesian optimization.

        This is a simplified version that focuses on exploring promising regions.
        """

        # Start with random initialization (10% of samples)
        init_samples = max(5, n_samples // 10)
        combinations = self._generate_random_combinations(init_samples)

        # For remaining samples, use focused exploration around promising regions
        # (This is simplified - real Bayesian optimization would use Gaussian Processes)

        for _ in range(n_samples - init_samples):
            param_dict = {}

            for param_name, param_values in self.param_dict.items():
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    # Focused exploration around existing good values
                    min_val, max_val = param_values

                    # Get existing values for this parameter
                    existing_values = [c[param_name] for c in combinations]

                    # Focus around the median of existing values with some exploration
                    focus_point = np.median(existing_values)
                    exploration_range = (max_val - min_val) * 0.3

                    new_value = np.random.normal(
                        focus_point,
                        exploration_range / 3
                    )

                    # Clamp to valid range
                    new_value = np.clip(new_value, min_val, max_val)

                    if param_name in self.continuous_params:
                        param_dict[param_name] = new_value
                    else:
                        param_dict[param_name] = int(np.round(new_value))

                else:
                    # Use weighted random choice based on frequency in existing combinations
                    param_dict[param_name] = np.random.choice(param_values)

            combinations.append(param_dict)

        self.combinations = combinations
        return combinations

    def get_n_combinations(self, strategy: str = 'grid', n_bins: int = 10) -> int:
        """Get number of combinations that would be generated."""
        if strategy == 'grid':
            total = 1
            for param_name, param_values in self.param_dict.items():
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous range
                    if param_name in self.continuous_params:
                        n_values = n_bins
                    else:
                        min_val, max_val = param_values
                        n_values = int(max_val) - int(min_val) + 1
                else:
                    # List of values
                    n_values = len(param_values)

                total *= n_values

            return total
        else:
            return self.param_dict.get('n_samples', 100)

    def validate_parameter_space(self) -> List[str]:
        """
        Validate parameter space for common issues.

        Returns:
            List of validation warnings
        """
        warnings_list = []

        for param_name, param_values in self.param_dict.items():
            if isinstance(param_values, tuple) and len(param_values) == 2:
                min_val, max_val = param_values
                if min_val >= max_val:
                    warnings_list.append(f"Parameter {param_name}: min_val >= max_val")
                elif min_val == max_val:
                    warnings_list.append(f"Parameter {param_name}: min_val == max_val (fixed value)")
            elif isinstance(param_values, list):
                if len(param_values) == 0:
                    warnings_list.append(f"Parameter {param_name}: empty value list")
                elif len(set(param_values)) != len(param_values):
                    warnings_list.append(f"Parameter {param_name}: duplicate values")

        return warnings_list


def create_normalization_parameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Create parameter grids for normalization methods."""
    return {
        'tmm_normalization': {
            'trim_ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
            'reference_sample': [None, 'auto']  # None means auto-detect
        },
        'sample_median_normalization': {
            'new_layer_name': ['sample_median_norm']  # Usually fixed, but included for completeness
        },
        'sample_mean_normalization': {
            'new_layer_name': ['sample_mean_norm']
        },
        'global_median_normalization': {
            'new_layer_name': ['global_median_norm']
        },
        'upper_quartile_normalization': {
            'percentile': [0.6, 0.7, 0.75, 0.8, 0.9],
            'new_layer_name': ['upper_quartile_norm']
        }
    }


def create_imputation_parameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Create parameter grids for imputation methods."""
    return {
        'knn': {
            'k': [3, 5, 7, 10, 15, 20],
            'new_layer_name': ['imputed']
        },
        'missforest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'new_layer_name': ['imputed']
        }
    }


def create_integration_parameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Create parameter grids for batch effect correction methods."""
    return {
        'combat': {
            'batch_key': ['batch'],  # Usually fixed
            'new_layer_name': ['combat_corrected']
        }
    }


def create_method_configs() -> Dict[str, MethodConfig]:
    """Create MethodConfig objects for common ScpTensor methods."""

    from scptensor.normalization import (
        tmm_normalization,
        sample_median_normalization,
        sample_mean_normalization,
        global_median_normalization,
        upper_quartile_normalization
    )
    from scptensor.impute import knn, missforest
    from scptensor.integration.combat import combat

    configs = {}

    # Normalization methods
    configs['tmm_normalization'] = MethodConfig(
        method_class=tmm_normalization,
        parameter_space=create_normalization_parameter_grids()['tmm_normalization'],
        default_parameters={'trim_ratio': 0.3, 'reference_sample': None},
        parameter_constraints={
            'trim_ratio': {'range': (0.0, 0.5), 'type': float},
            'reference_sample': {'choices': [None, 'auto'], 'type': (type(None), str)}
        }
    )

    configs['sample_median_normalization'] = MethodConfig(
        method_class=sample_median_normalization,
        default_parameters={'new_layer_name': 'sample_median_norm'}
    )

    configs['sample_mean_normalization'] = MethodConfig(
        method_class=sample_mean_normalization,
        default_parameters={'new_layer_name': 'sample_mean_norm'}
    )

    configs['global_median_normalization'] = MethodConfig(
        method_class=global_median_normalization,
        default_parameters={'new_layer_name': 'global_median_norm'}
    )

    configs['upper_quartile_normalization'] = MethodConfig(
        method_class=upper_quartile_normalization,
        parameter_space=create_normalization_parameter_grids()['upper_quartile_normalization'],
        default_parameters={'percentile': 0.75, 'new_layer_name': 'upper_quartile_norm'},
        parameter_constraints={
            'percentile': {'range': (0.5, 0.95), 'type': float}
        }
    )

    # Imputation methods
    configs['knn'] = MethodConfig(
        method_class=knn,
        parameter_space=create_imputation_parameter_grids()['knn'],
        default_parameters={'k': 5, 'new_layer_name': 'imputed'},
        parameter_constraints={
            'k': {'range': (1, 50), 'type': int}
        }
    )

    # Integration methods
    configs['combat'] = MethodConfig(
        method_class=combat,
        parameter_space=create_integration_parameter_grids()['combat'],
        default_parameters={'batch_key': 'batch', 'new_layer_name': 'combat_corrected'}
    )

    return configs