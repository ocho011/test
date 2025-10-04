"""Configuration manager for loading and managing YAML configurations."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from .models import TradingBotConfig
from .yaml_loader import load_yaml_with_env


class ConfigManager:
    """
    Manages YAML-based configuration files with environment support.
    
    Features:
    - Environment-based config loading (development, production, testing)
    - Pydantic validation and type checking
    - Configuration caching
    - Environment variable override
    - Hot-reload capability
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[TradingBotConfig] = None
    _config_path: Optional[Path] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config manager."""
        self._cache_enabled = True
    
    def load_config(
        self, 
        config_path: Optional[str] = None,
        environment: Optional[str] = None
    ) -> TradingBotConfig:
        """
        Load configuration from YAML file with inheritance support.
        
        Args:
            config_path: Path to config file. If None, auto-detect from environment
            environment: Environment name (production, paper-mainnet, paper-testnet, development)
        
        Returns:
            TradingBotConfig: Validated configuration object
        """
        # Determine environment
        if environment is None:
            environment = os.getenv("TRADING_ENV", "development")
        
        # Determine config path
        if config_path is None:
            config_dir = Path(__file__).parent / "environments"
            config_path = config_dir / f"{environment}.yml"
        else:
            config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML with inheritance and environment variable substitution
        config_data = self._load_yaml_with_inheritance(config_path)
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Validate with pydantic
        self._config = TradingBotConfig(**config_data)
        self._config_path = config_path
        
        return self._config

    
    def _load_yaml_with_inheritance(self, config_path: Path) -> Dict[str, Any]:
        """
        Load YAML file with inheritance support.
        
        Supports 'extends' key to inherit from base configuration.
        Base config is loaded first, then environment-specific config is deep-merged.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dict containing merged configuration with env vars substituted
        """
        # Load the environment-specific config
        with open(config_path, 'r', encoding='utf-8') as f:
            env_config = load_yaml_with_env(f)
        
        # Check if config extends another file
        if 'extends' not in env_config:
            return env_config
        
        # Get base config path
        base_file = env_config.pop('extends')
        base_path = config_path.parent / base_file
        
        # Check if base file exists
        if not base_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_path}")
        
        # Load base config recursively (to support multi-level inheritance)
        base_config = self._load_yaml_with_inheritance(base_path)
        
        # Deep merge: environment config overrides base config
        merged_config = self._deep_merge(base_config, env_config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Override values take precedence over base values.
        Nested dictionaries are merged recursively.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def _load_yaml_with_env_substitution(self, config_path: Path) -> Dict[str, Any]:
        """
        Load YAML file with environment variable substitution.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dict containing parsed YAML with env vars substituted
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return load_yaml_with_env(f)
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config.
        
        Environment variables follow pattern: TRADING_<SECTION>_<KEY>
        Example: TRADING_BINANCE_API_KEY overrides binance.api_key
        """
        # Binance overrides
        if api_key := os.getenv("TRADING_BINANCE_API_KEY"):
            config_data.setdefault("binance", {})["api_key"] = api_key
        if api_secret := os.getenv("TRADING_BINANCE_API_SECRET"):
            config_data.setdefault("binance", {})["api_secret"] = api_secret
        
        # Discord overrides
        if bot_token := os.getenv("TRADING_DISCORD_BOT_TOKEN"):
            config_data.setdefault("discord", {})["bot_token"] = bot_token
        if channel_id := os.getenv("TRADING_DISCORD_CHANNEL_ID"):
            config_data.setdefault("discord", {})["channel_id"] = int(channel_id)
        
        # System overrides
        if debug := os.getenv("TRADING_DEBUG"):
            config_data.setdefault("system", {})["debug"] = debug.lower() == "true"
        
        # Logging overrides
        if log_level := os.getenv("TRADING_LOG_LEVEL"):
            config_data.setdefault("logging", {})["level"] = log_level.upper()
        
        return config_data
    
    def get_config(self) -> TradingBotConfig:
        """
        Get current configuration.
        
        Returns:
            TradingBotConfig: Current configuration
            
        Raises:
            RuntimeError: If config not loaded yet
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def reload_config(self) -> TradingBotConfig:
        """
        Reload configuration from file (hot-reload).
        
        Returns:
            TradingBotConfig: Reloaded configuration
        """
        if self._config_path is None:
            raise RuntimeError("No config path available for reload")
        
        return self.load_config(str(self._config_path))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., "binance.api_key")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        config = self.get_config()
        keys = key.split(".")
        value = config.model_dump()
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key (runtime only).
        
        Args:
            key: Dot-notation key (e.g., "system.debug")
            value: New value
        """
        config = self.get_config()
        keys = key.split(".")
        
        # Navigate to the target
        target = config
        for k in keys[:-1]:
            target = getattr(target, k)
        
        # Set the value
        setattr(target, keys[-1], value)
    
    @lru_cache(maxsize=128)
    def get_cached(self, key: str) -> Any:
        """
        Get cached configuration value.
        
        Args:
            key: Dot-notation key
        
        Returns:
            Cached configuration value
        """
        return self.get(key)
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.get_cached.cache_clear()

    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment configuration information.
        
        Returns information about the current environment including:
        - Environment name
        - Binance network (mainnet/testnet)
        - Execution mode (live/simulation)
        - Dry-run status
        
        Returns:
            Dict containing environment information
        """
        config = self.get_config()
        
        # Determine execution mode based on dry_run setting
        dry_run = config.custom.get('dry_run', False)
        execution_mode = 'simulation' if dry_run else 'live'
        
        # Determine binance network
        binance_network = 'testnet' if config.binance.testnet else 'mainnet'
        
        return {
            'environment': config.system.environment,
            'binance_network': binance_network,
            'execution_mode': execution_mode,
            'dry_run': dry_run,
            'testnet': config.binance.testnet,
            'description': self._get_environment_description(
                config.system.environment,
                binance_network,
                execution_mode
            )
        }
    
    @staticmethod
    def _get_environment_description(env: str, network: str, mode: str) -> str:
        """
        Get human-readable description of environment configuration.
        
        Args:
            env: Environment name
            network: Binance network (mainnet/testnet)
            mode: Execution mode (live/simulation)
        
        Returns:
            Human-readable description
        """
        descriptions = {
            ('production', 'mainnet', 'live'): 'Production: Live trading on Binance mainnet',
            ('paper-mainnet', 'mainnet', 'simulation'): 'Paper-Mainnet: Simulated trading with mainnet data',
            ('paper-testnet', 'testnet', 'simulation'): 'Paper-Testnet: Simulated trading with testnet data',
            ('development', 'testnet', 'live'): 'Development: Live trading on Binance testnet',
        }
        
        key = (env, network, mode)
        return descriptions.get(key, f'{env.title()}: {network} + {mode}')
    
    @staticmethod
    def list_available_environments() -> Dict[str, Dict[str, str]]:
        """
        List all available standard environments with their configurations.
        
        Returns:
            Dict mapping environment names to their binance/execution settings
        """
        return {
            'production': {
                'binance_network': 'mainnet',
                'execution_mode': 'live',
                'dry_run': False,
                'description': 'Production: Live trading on Binance mainnet'
            },
            'paper-mainnet': {
                'binance_network': 'mainnet',
                'execution_mode': 'simulation',
                'dry_run': True,
                'description': 'Paper-Mainnet: Simulated trading with mainnet data'
            },
            'paper-testnet': {
                'binance_network': 'testnet',
                'execution_mode': 'simulation',
                'dry_run': True,
                'description': 'Paper-Testnet: Simulated trading with testnet data'
            },
            'development': {
                'binance_network': 'testnet',
                'execution_mode': 'live',
                'dry_run': False,
                'description': 'Development: Live trading on Binance testnet'
            }
        }


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> TradingBotConfig:
    """
    Get global configuration instance.
    
    Returns:
        TradingBotConfig: Global configuration
    """
    return config_manager.get_config()


def load_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None
) -> TradingBotConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        environment: Environment name
    
    Returns:
        TradingBotConfig: Loaded configuration
    """
    return config_manager.load_config(config_path, environment)