"""Custom YAML loader with environment variable substitution."""
import os
import re
import yaml
from typing import Any, Dict, Set, Optional


class CircularReferenceError(Exception):
    """Raised when circular reference is detected in env var substitution."""
    pass


class EnvVarNotFoundError(Exception):
    """Raised when required environment variable is not found."""
    pass


class CustomYAMLLoader(yaml.SafeLoader):
    """
    Extended SafeLoader with environment variable substitution support.
    
    Supports patterns:
    - ${ENV_VAR} - Required env var, raises error if not found
    - ${ENV_VAR:default_value} - Optional env var with default value
    
    Features:
    - Type preservation (string, int, float, bool)
    - Nested substitution prevention (depth tracking)
    - Circular reference detection
    - Proper error handling
    """
    
    # Pattern for matching ${ENV_VAR} or ${ENV_VAR:default}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
    MAX_SUBSTITUTION_DEPTH = 10
    
    def __init__(self, stream):
        """Initialize loader with tracking for circular references."""
        super().__init__(stream)
        self._substitution_stack: Set[str] = set()
        self._substitution_depth = 0
    
    def construct_scalar(self, node):
        """Override scalar construction to support env var substitution."""
        value = super().construct_scalar(node)
        
        # Only process strings
        if not isinstance(value, str):
            return value
        
        # Check if contains env var pattern
        if not self.ENV_VAR_PATTERN.search(value):
            return value
        
        # Perform substitution
        return self._substitute_env_vars(value)
    
    def _substitute_env_vars(self, value: str) -> Any:
        """
        Substitute environment variables in value string.
        
        Args:
            value: String containing ${ENV_VAR} patterns
            
        Returns:
            Substituted value with type conversion
            
        Raises:
            CircularReferenceError: If circular reference detected
            EnvVarNotFoundError: If required env var not found
        """
        # Check depth limit to prevent infinite recursion
        self._substitution_depth += 1
        if self._substitution_depth > self.MAX_SUBSTITUTION_DEPTH:
            raise CircularReferenceError(
                f"Maximum substitution depth ({self.MAX_SUBSTITUTION_DEPTH}) exceeded. "
                "Possible circular reference in environment variables."
            )
        
        try:
            # Find all env var patterns
            matches = list(self.ENV_VAR_PATTERN.finditer(value))
            
            if not matches:
                return value
            
            # If the entire value is a single env var, preserve type
            if len(matches) == 1 and matches[0].group(0) == value:
                env_var = matches[0].group(1)
                default_value = matches[0].group(2)
                
                # Check for circular reference
                if env_var in self._substitution_stack:
                    raise CircularReferenceError(
                        f"Circular reference detected: {env_var} -> {' -> '.join(self._substitution_stack)}"
                    )
                
                # Get env var value
                env_value = os.getenv(env_var)
                
                if env_value is None:
                    if default_value is not None:
                        result = default_value
                    else:
                        raise EnvVarNotFoundError(
                            f"Environment variable '{env_var}' not found and no default provided"
                        )
                else:
                    result = env_value
                
                # Type conversion for single values
                return self._convert_type(result)
            
            # Multiple substitutions or partial string - return as string
            result = value
            for match in reversed(matches):  # Reverse to maintain string indices
                env_var = match.group(1)
                default_value = match.group(2)
                
                # Check for circular reference
                if env_var in self._substitution_stack:
                    raise CircularReferenceError(
                        f"Circular reference detected: {env_var}"
                    )
                
                # Get env var value
                env_value = os.getenv(env_var)
                
                if env_value is None:
                    if default_value is not None:
                        replacement = default_value
                    else:
                        raise EnvVarNotFoundError(
                            f"Environment variable '{env_var}' not found and no default provided"
                        )
                else:
                    replacement = env_value
                
                # Replace in string
                result = result[:match.start()] + replacement + result[match.end():]
            
            return result
        
        finally:
            self._substitution_depth -= 1
    
    def _convert_type(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try boolean
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        if value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Try integer
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value


def load_yaml_with_env(stream) -> Dict[str, Any]:
    """
    Load YAML file with environment variable substitution.
    
    Args:
        stream: File stream or string to load
        
    Returns:
        Parsed YAML data with env vars substituted
    """
    return yaml.load(stream, Loader=CustomYAMLLoader)
