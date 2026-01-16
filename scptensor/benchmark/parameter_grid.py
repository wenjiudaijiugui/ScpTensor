"""
Parameter grid system for systematic parameter exploration and optimization.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MethodConfig:
    """Configuration for a specific method.

    Attributes
    ----------
    method_class : Any
        The actual method function/class (must be callable).
    parameter_space : dict[str, list[Any]]
        Valid parameter values for grid search.
    default_parameters : dict[str, Any]
        Default parameter values.
    parameter_constraints : dict[str, Any]
        Validation constraints for parameters.
    """

    method_class: Any
    parameter_space: dict[str, list[Any]] = field(default_factory=dict)
    default_parameters: dict[str, Any] = field(default_factory=dict)
    parameter_constraints: dict[str, Any] = field(default_factory=dict)

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate parameters against constraints.

        Parameters
        ----------
        parameters : dict[str, Any]
            Parameters to validate.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        ValueError
            If parameters violate constraints.
        """
        for name, value in parameters.items():
            if name not in self.parameter_constraints:
                continue

            constraint = self.parameter_constraints[name]
            self._validate_single(name, value, constraint)

        return True

    def _validate_single(self, name: str, value: Any, constraint: dict[str, Any]) -> None:
        """Validate a single parameter against constraints.

        Parameters
        ----------
        name : str
            Parameter name.
        value : Any
            Parameter value.
        constraint : dict[str, Any]
            Constraint specification.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if "type" in constraint:
            expected_type = constraint["type"]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Parameter {name} must be of type {expected_type}, got {type(value)}"
                )

        if "range" in constraint:
            min_val, max_val = constraint["range"]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Parameter {name}={value} out of range [{min_val}, {max_val}]")

        if "choices" in constraint and value not in constraint["choices"]:
            raise ValueError(f"Parameter {name}={value} not in choices {constraint['choices']}")

        if "min" in constraint and value < constraint["min"]:
            raise ValueError(f"Parameter {name}={value} below minimum {constraint['min']}")

        if "max" in constraint and value > constraint["max"]:
            raise ValueError(f"Parameter {name}={value} above maximum {constraint['max']}")


class ParameterGrid:
    """Manages parameter combinations for systematic exploration.

    Supports multiple generation strategies:
    - Grid search: Exhaustive combination of all parameter values
    - Random search: Random sampling of parameter combinations
    - Bayesian optimization: Smart parameter selection (simplified)

    Attributes
    ----------
    param_dict : dict[str, list[Any] | tuple[float, float]]
        Parameter definitions.
    continuous_params : list[str]
        Names of continuous parameters.
    combinations : list[dict[str, Any]]
        Cached generated combinations.
    """

    __slots__ = ("param_dict", "continuous_params", "combinations")

    def __init__(
        self,
        param_dict: dict[str, list[Any] | tuple[float, float]],
        continuous_params: list[str] | None = None,
    ) -> None:
        """Initialize parameter grid.

        Parameters
        ----------
        param_dict : dict[str, list[Any] | tuple[float, float]]
            Dictionary mapping parameter names to values or (min, max) ranges.
        continuous_params : list[str] | None
            Names of parameters that are continuous.
        """
        self.param_dict = param_dict
        self.continuous_params = continuous_params or []
        self.combinations: list[dict[str, Any]] = []

    def generate_combinations(
        self,
        strategy: str = "grid",
        n_samples: int = 100,
        n_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate parameter combinations.

        Parameters
        ----------
        strategy : str
            Generation strategy: 'grid', 'random', or 'bayesian'.
        n_samples : int
            Number of samples for random/bayesian strategies.
        n_bins : int
            Number of bins for discretizing continuous ranges in grid search.

        Returns
        -------
        list[dict[str, Any]]
            List of parameter dictionaries.

        Raises
        ------
        ValueError
            If strategy is unknown.
        """
        generators = {
            "grid": lambda: self._generate_grid(n_bins),
            "random": lambda: self._generate_random(n_samples),
            "bayesian": lambda: self._generate_bayesian(n_samples),
        }

        if strategy not in generators:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(generators)}")

        self.combinations = generators[strategy]()
        return self.combinations

    def _generate_grid(self, n_bins: int) -> list[dict[str, Any]]:
        """Generate exhaustive grid search combinations.

        Parameters
        ----------
        n_bins : int
            Number of bins for continuous parameter discretization.

        Returns
        -------
        list[dict[str, Any]]
            All parameter combinations.
        """
        param_lists = {}

        for name, values in self.param_dict.items():
            if isinstance(values, tuple) and len(values) == 2:
                min_val, max_val = values
                if name in self.continuous_params:
                    param_lists[name] = np.linspace(min_val, max_val, n_bins).tolist()
                else:
                    param_lists[name] = list(range(int(min_val), int(max_val) + 1))
            else:
                param_lists[name] = values

        keys = list(param_lists.keys())
        values = list(param_lists.values())

        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]

    def _generate_random(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate random parameter combinations.

        Parameters
        ----------
        n_samples : int
            Number of random samples to generate.

        Returns
        -------
        list[dict[str, Any]]
            Random parameter combinations.
        """
        combinations = []

        for _ in range(n_samples):
            param_dict = {}

            for name, values in self.param_dict.items():
                if isinstance(values, tuple) and len(values) == 2:
                    min_val, max_val = values
                    if name in self.continuous_params:
                        param_dict[name] = np.random.uniform(min_val, max_val)
                    else:
                        param_dict[name] = np.random.randint(int(min_val), int(max_val) + 1)
                else:
                    param_dict[name] = np.random.choice(values)

            combinations.append(param_dict)

        return combinations

    def _generate_bayesian(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate combinations using simplified Bayesian optimization.

        Focuses exploration around promising regions.

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.

        Returns
        -------
        list[dict[str, Any]]
            Parameter combinations with focused exploration.
        """
        # Initialize with 10% random samples
        init_samples = max(5, n_samples // 10)
        combinations = self._generate_random(init_samples)

        # Focus exploration around existing values
        for _ in range(n_samples - init_samples):
            param_dict = {}

            for name, values in self.param_dict.items():
                if isinstance(values, tuple) and len(values) == 2:
                    min_val, max_val = values
                    existing = [c[name] for c in combinations]

                    # Explore around median
                    focus = np.median(existing)
                    exploration_range = (max_val - min_val) * 0.3
                    new_val = np.clip(
                        np.random.normal(focus, exploration_range / 3),
                        min_val,
                        max_val,
                    )

                    param_dict[name] = (
                        new_val if name in self.continuous_params else int(np.round(new_val))
                    )
                else:
                    param_dict[name] = np.random.choice(values)

            combinations.append(param_dict)

        return combinations

    def get_n_combinations(self, strategy: str = "grid", n_bins: int = 10) -> int:
        """Get number of combinations that would be generated.

        Parameters
        ----------
        strategy : str
            Generation strategy.
        n_bins : int
            Number of bins for grid search.

        Returns
        -------
        int
            Number of combinations.
        """
        if strategy != "grid":
            return 100

        total = 1
        for name, values in self.param_dict.items():
            if isinstance(values, tuple) and len(values) == 2:
                if name in self.continuous_params:
                    n_values = n_bins
                else:
                    min_val, max_val = values
                    n_values = int(max_val) - int(min_val) + 1
            else:
                n_values = len(values)

            total *= n_values

        return total

    def validate_parameter_space(self) -> list[str]:
        """Validate parameter space for common issues.

        Returns
        -------
        list[str]
            List of validation warnings.
        """
        warnings_list = []

        for name, values in self.param_dict.items():
            if isinstance(values, tuple) and len(values) == 2:
                min_val, max_val = values
                if min_val >= max_val:
                    warnings_list.append(f"Parameter {name}: min_val >= max_val")
                elif min_val == max_val:
                    warnings_list.append(f"Parameter {name}: min_val == max_val (fixed value)")
            elif isinstance(values, list):
                if not values:
                    warnings_list.append(f"Parameter {name}: empty value list")
                elif len(set(values)) != len(values):
                    warnings_list.append(f"Parameter {name}: duplicate values")

        return warnings_list


def create_normalization_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Create parameter grids for normalization methods.

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Parameter grids for normalization methods.
    """
    return {
        "tmm_normalization": {
            "trim_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
            "reference_sample": [None, "auto"],
        },
        "sample_median_normalization": {"new_layer_name": ["sample_median_norm"]},
        "sample_mean_normalization": {"new_layer_name": ["sample_mean_norm"]},
        "global_median_normalization": {"new_layer_name": ["global_median_norm"]},
        "upper_quartile_normalization": {
            "percentile": [0.6, 0.7, 0.75, 0.8, 0.9],
            "new_layer_name": ["upper_quartile_norm"],
        },
    }


def create_imputation_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Create parameter grids for imputation methods.

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Parameter grids for imputation methods.
    """
    return {
        "knn": {"k": [3, 5, 7, 10, 15, 20], "new_layer_name": ["imputed"]},
        "missforest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "new_layer_name": ["imputed"],
        },
    }


def create_integration_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Create parameter grids for batch effect correction methods.

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Parameter grids for integration methods.
    """
    return {
        "combat": {
            "batch_key": ["batch"],
            "new_layer_name": ["combat_corrected"],
        }
    }


def create_method_configs() -> dict[str, MethodConfig]:
    """Create MethodConfig objects for common ScpTensor methods.

    Returns
    -------
    dict[str, MethodConfig]
        Method configurations.
    """
    from scptensor.impute import impute_knn
    from scptensor.integration.combat import integrate_combat
    from scptensor.normalization import (
        norm_global_median,
        norm_quartile,
        norm_sample_mean,
        norm_sample_median,
        norm_tmm,
    )

    norm_grids = create_normalization_parameter_grids()
    impute_grids = create_imputation_parameter_grids()
    integration_grids = create_integration_parameter_grids()

    return {
        "tmm_normalization": MethodConfig(
            method_class=norm_tmm,
            parameter_space=norm_grids["tmm_normalization"],
            default_parameters={"trim_ratio": 0.3, "reference_sample": None},
            parameter_constraints={
                "trim_ratio": {"range": (0.0, 0.5), "type": float},
                "reference_sample": {"choices": [None, "auto"], "type": (type(None), str)},
            },
        ),
        "sample_median_normalization": MethodConfig(
            method_class=norm_sample_median,
            default_parameters={"new_layer_name": "sample_median_norm"},
        ),
        "sample_mean_normalization": MethodConfig(
            method_class=norm_sample_mean,
            default_parameters={"new_layer_name": "sample_mean_norm"},
        ),
        "global_median_normalization": MethodConfig(
            method_class=norm_global_median,
            default_parameters={"new_layer_name": "global_median_norm"},
        ),
        "upper_quartile_normalization": MethodConfig(
            method_class=norm_quartile,
            parameter_space=norm_grids["upper_quartile_normalization"],
            default_parameters={"percentile": 0.75, "new_layer_name": "upper_quartile_norm"},
            parameter_constraints={"percentile": {"range": (0.5, 0.95), "type": float}},
        ),
        "knn": MethodConfig(
            method_class=impute_knn,
            parameter_space=impute_grids["knn"],
            default_parameters={"k": 5, "new_layer_name": "imputed"},
            parameter_constraints={"k": {"range": (1, 50), "type": int}},
        ),
        "combat": MethodConfig(
            method_class=integrate_combat,
            parameter_space=integration_grids["combat"],
            default_parameters={"batch_key": "batch", "new_layer_name": "combat_corrected"},
        ),
    }
