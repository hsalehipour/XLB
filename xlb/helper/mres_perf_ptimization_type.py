import argparse
from enum import Enum


class MresPerfOptimizationType(Enum):
    """
    Enumeration of available optimization strategies for the LBM solver.

    Supports parsing from either the enum member name (case-insensitive)
    or its integer value, and provides a method to build the CLI parser.
    """
    NAIVE_COLLIDE_STREAM = 0
    FUSION_AT_FINEST = 1

    @staticmethod
    def from_string(value: str) -> "MresPerfOptimizationType":
        """
        Parse a string to an OptimizationType.

        Accepts either the enum member name (case-insensitive) or its integer value.

        Args:
            value: The enum name (e.g. 'naive_collide_stream') or integer value (e.g. '0').

        Returns:
            An OptimizationType member.

        Raises:
            argparse.ArgumentTypeError: If the input is invalid.
        """
        # Attempt to parse by name (case-insensitive)
        key = value.strip().upper()
        if key in MresPerfOptimizationType.__members__:
            return MresPerfOptimizationType[key]

        # Attempt to parse by integer value
        try:
            int_value = int(value)
            return MresPerfOptimizationType(int_value)
        except (ValueError, KeyError):
            valid_options = ", ".join(f"{member.name}({member.value})" for member in MresPerfOptimizationType)
            raise argparse.ArgumentTypeError(
                f"Invalid OptimizationType {value!r}. Choose from: {valid_options}."
            )

    def __str__(self) -> str:
        """
        Return a human-readable string for the enum member.
        """
        return self.name

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        """
        Create and configure the argument parser with optimization option.

        Returns:
            A configured ArgumentParser instance.
        """
        parser = argparse.ArgumentParser(
            description="Run the LBM multiresolution simulation with specified optimizations."
        )
        # Dynamically generate help text from enum members
        valid_options = ", ".join(f"{member.name}({member.value})" for member in MresPerfOptimizationType)
        parser.add_argument(
            "-o", "--optimization",
            type=MresPerfOptimizationType.from_string,
            default=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
            help=f"Select optimization strategy: {valid_options}",
        )
        return parser
