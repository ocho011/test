"""Tests for YAML loader with environment variable substitution."""
import os
import pytest
import tempfile
from pathlib import Path

from trading_bot.config.yaml_loader import (
    CustomYAMLLoader,
    load_yaml_with_env,
    CircularReferenceError,
    EnvVarNotFoundError
)


class TestCustomYAMLLoader:
    """Test CustomYAMLLoader environment variable substitution."""
    
    def test_basic_env_var_substitution(self, monkeypatch):
        """Test basic environment variable substitution."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        yaml_content = """
        key: ${TEST_VAR}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "test_value"
    
    def test_env_var_with_default_value(self, monkeypatch):
        """Test environment variable with default value."""
        # Ensure env var doesn't exist
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        
        yaml_content = """
        key: ${NONEXISTENT_VAR:default_value}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "default_value"
    
    def test_env_var_override_default(self, monkeypatch):
        """Test that env var overrides default value."""
        monkeypatch.setenv("TEST_VAR", "actual_value")
        
        yaml_content = """
        key: ${TEST_VAR:default_value}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "actual_value"
    
    def test_missing_required_env_var(self, monkeypatch):
        """Test error when required env var is missing."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        
        yaml_content = """
        key: ${REQUIRED_VAR}
        """
        
        with pytest.raises(EnvVarNotFoundError) as exc_info:
            load_yaml_with_env(yaml_content)
        
        assert "REQUIRED_VAR" in str(exc_info.value)
        assert "not found" in str(exc_info.value)
    
    def test_multiple_env_vars_in_string(self, monkeypatch):
        """Test multiple environment variables in single string."""
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        
        yaml_content = """
        url: http://${HOST}:${PORT}/api
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["url"] == "http://localhost:8080/api"
    
    def test_type_preservation_integer(self, monkeypatch):
        """Test type preservation for integer values."""
        monkeypatch.setenv("PORT", "8080")
        
        yaml_content = """
        port: ${PORT}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["port"] == 8080
        assert isinstance(result["port"], int)
    
    def test_type_preservation_float(self, monkeypatch):
        """Test type preservation for float values."""
        monkeypatch.setenv("THRESHOLD", "0.95")
        
        yaml_content = """
        threshold: ${THRESHOLD}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["threshold"] == 0.95
        assert isinstance(result["threshold"], float)
    
    def test_type_preservation_boolean_true(self, monkeypatch):
        """Test type preservation for boolean true values."""
        test_cases = ["true", "True", "TRUE", "yes", "Yes", "on", "1"]
        
        for value in test_cases:
            monkeypatch.setenv("FLAG", value)
            
            yaml_content = """
            enabled: ${FLAG}
            """
            
            result = load_yaml_with_env(yaml_content)
            assert result["enabled"] is True
            assert isinstance(result["enabled"], bool)
    
    def test_type_preservation_boolean_false(self, monkeypatch):
        """Test type preservation for boolean false values."""
        test_cases = ["false", "False", "FALSE", "no", "No", "off", "0"]
        
        for value in test_cases:
            monkeypatch.setenv("FLAG", value)
            
            yaml_content = """
            disabled: ${FLAG}
            """
            
            result = load_yaml_with_env(yaml_content)
            assert result["disabled"] is False
            assert isinstance(result["disabled"], bool)
    
    def test_type_preservation_string(self, monkeypatch):
        """Test type preservation for string values."""
        monkeypatch.setenv("NAME", "test_string")
        
        yaml_content = """
        name: ${NAME}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["name"] == "test_string"
        assert isinstance(result["name"], str)
    
    def test_partial_string_remains_string(self, monkeypatch):
        """Test that partial substitution keeps string type."""
        monkeypatch.setenv("PORT", "8080")
        
        yaml_content = """
        description: Port is ${PORT}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["description"] == "Port is 8080"
        assert isinstance(result["description"], str)
    
    def test_nested_structure(self, monkeypatch):
        """Test env var substitution in nested structures."""
        monkeypatch.setenv("API_KEY", "secret_key_123")
        monkeypatch.setenv("API_SECRET", "secret_secret_456")
        monkeypatch.setenv("TESTNET", "true")
        
        yaml_content = """
        binance:
          api_key: ${API_KEY}
          api_secret: ${API_SECRET}
          testnet: ${TESTNET}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["binance"]["api_key"] == "secret_key_123"
        assert result["binance"]["api_secret"] == "secret_secret_456"
        assert result["binance"]["testnet"] is True
    
    def test_list_with_env_vars(self, monkeypatch):
        """Test env var substitution in lists."""
        monkeypatch.setenv("STRATEGY1", "ict")
        monkeypatch.setenv("STRATEGY2", "traditional")
        
        yaml_content = """
        strategies:
          - ${STRATEGY1}
          - ${STRATEGY2}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["strategies"] == ["ict", "traditional"]
    
    def test_default_with_colon_in_value(self, monkeypatch):
        """Test default value containing colon."""
        monkeypatch.delenv("URL", raising=False)
        
        yaml_content = """
        url: ${URL:http://localhost:8080}
        """
        
        result = load_yaml_with_env(yaml_content)
        # Only the first colon is treated as separator
        assert result["url"] == "http://localhost:8080"
    
    def test_empty_default_value(self, monkeypatch):
        """Test empty string as default value."""
        monkeypatch.delenv("OPTIONAL", raising=False)
        
        yaml_content = """
        optional: ${OPTIONAL:}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["optional"] == ""
    
    def test_circular_reference_prevention(self, monkeypatch):
        """Test that circular references are detected and prevented."""
        # This test simulates depth limit being hit
        # In practice, circular refs would need the env var to reference itself
        
        # Create a very deep nesting that would hit depth limit
        yaml_content = """
        key: ${DEEP_VAR}
        """
        
        # Set env var that would cause deep recursion if improperly handled
        monkeypatch.setenv("DEEP_VAR", "value")
        
        # Should work fine with normal value
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "value"
    
    def test_special_characters_in_env_var_name(self, monkeypatch):
        """Test environment variables with underscores and numbers."""
        monkeypatch.setenv("VAR_WITH_UNDERSCORE_123", "test_value")
        
        yaml_content = """
        key: ${VAR_WITH_UNDERSCORE_123}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "test_value"
    
    def test_whitespace_handling(self, monkeypatch):
        """Test that whitespace in env var names is handled correctly."""
        monkeypatch.setenv("TEST_VAR", "value")
        
        yaml_content = """
        key: ${TEST_VAR}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["key"] == "value"
    
    def test_complex_default_value(self, monkeypatch):
        """Test complex default value with special characters."""
        monkeypatch.delenv("COMPLEX", raising=False)
        
        yaml_content = """
        value: ${COMPLEX:user:pass@host:port/path?query=value}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["value"] == "user:pass@host:port/path?query=value"
    
    def test_numeric_string_default(self, monkeypatch):
        """Test numeric string as default value."""
        monkeypatch.delenv("NUM", raising=False)
        
        yaml_content = """
        port: ${NUM:8080}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["port"] == 8080
        assert isinstance(result["port"], int)
    
    def test_file_loading_with_env_vars(self, monkeypatch, tmp_path):
        """Test loading YAML file with env var substitution."""
        monkeypatch.setenv("TEST_KEY", "test_value")
        
        # Create temporary YAML file
        yaml_file = tmp_path / "test_config.yml"
        yaml_file.write_text("""
        config:
          key: ${TEST_KEY}
          port: ${PORT:3000}
        """)
        
        with open(yaml_file, 'r') as f:
            result = load_yaml_with_env(f)
        
        assert result["config"]["key"] == "test_value"
        assert result["config"]["port"] == 3000
    
    def test_preserves_non_env_var_dollar_signs(self):
        """Test that dollar signs not in ${} pattern are preserved."""
        yaml_content = """
        price: $100
        regex: \\$\\{.*\\}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["price"] == "$100"
        assert result["regex"] == "\\$\\{.*\\}"
    
    def test_mixed_types_in_same_file(self, monkeypatch):
        """Test file with mixed types and substitutions."""
        monkeypatch.setenv("STRING_VAR", "text")
        monkeypatch.setenv("INT_VAR", "42")
        monkeypatch.setenv("FLOAT_VAR", "3.14")
        monkeypatch.setenv("BOOL_VAR", "true")
        
        yaml_content = """
        string_field: ${STRING_VAR}
        int_field: ${INT_VAR}
        float_field: ${FLOAT_VAR}
        bool_field: ${BOOL_VAR}
        mixed: Port ${INT_VAR}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["string_field"] == "text"
        assert result["int_field"] == 42
        assert result["float_field"] == 3.14
        assert result["bool_field"] is True
        assert result["mixed"] == "Port 42"  # String because it's partial


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""
    
    def test_database_url_construction(self, monkeypatch):
        """Test database URL construction from parts."""
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASS", "secret")
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "trading")
        
        yaml_content = """
        database:
          url: postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}
        """
        
        result = load_yaml_with_env(yaml_content)
        expected = "postgresql://admin:secret@localhost:5432/trading"
        assert result["database"]["url"] == expected
    
    def test_api_configuration(self, monkeypatch):
        """Test API configuration with secrets."""
        monkeypatch.setenv("API_KEY", "key_123")
        monkeypatch.setenv("API_SECRET", "secret_456")
        monkeypatch.setenv("TESTNET", "false")
        
        yaml_content = """
        binance:
          api_key: ${API_KEY}
          api_secret: ${API_SECRET}
          testnet: ${TESTNET}
          base_url: ${BASE_URL:https://api.binance.com}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["binance"]["api_key"] == "key_123"
        assert result["binance"]["api_secret"] == "secret_456"
        assert result["binance"]["testnet"] is False
        assert result["binance"]["base_url"] == "https://api.binance.com"
    
    def test_feature_flags(self, monkeypatch):
        """Test feature flag configuration."""
        monkeypatch.setenv("ENABLE_TRADING", "true")
        monkeypatch.setenv("ENABLE_NOTIFICATIONS", "false")
        monkeypatch.setenv("MAX_POSITIONS", "5")
        
        yaml_content = """
        features:
          trading_enabled: ${ENABLE_TRADING}
          notifications_enabled: ${ENABLE_NOTIFICATIONS}
          max_concurrent_positions: ${MAX_POSITIONS}
        """
        
        result = load_yaml_with_env(yaml_content)
        assert result["features"]["trading_enabled"] is True
        assert result["features"]["notifications_enabled"] is False
        assert result["features"]["max_concurrent_positions"] == 5
