"""
Comprehensive tests for Vultr VM deployment script.

These tests verify the deployment script functionality including:
- Script validation and syntax checking
- Function existence and structure
- Configuration handling
- Error handling and rollback mechanisms
- System requirements checking
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


class TestDeploymentScriptStructure(unittest.TestCase):
    """Test the structure and syntax of the deployment script."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        cls.assertTrue(cls.script_path.exists(), f"Deployment script not found at {cls.script_path}")
    
    def test_script_exists(self):
        """Test that the deployment script exists."""
        self.assertTrue(self.script_path.exists())
    
    def test_script_is_executable(self):
        """Test that the script has executable permissions."""
        self.assertTrue(os.access(self.script_path, os.X_OK))
    
    def test_script_has_shebang(self):
        """Test that the script has a proper shebang."""
        with open(self.script_path, 'r') as f:
            first_line = f.readline().strip()
            self.assertEqual(first_line, "#!/bin/bash")
    
    def test_script_syntax_valid(self):
        """Test that the script has valid bash syntax."""
        result = subprocess.run(
            ["bash", "-n", str(self.script_path)],
            capture_output=True,
            text=True
        )
        self.assertEqual(
            result.returncode, 0,
            f"Script has syntax errors:\n{result.stderr}"
        )
    
    def test_required_functions_exist(self):
        """Test that all required functions are defined in the script."""
        required_functions = [
            "print_info",
            "print_warn",
            "print_error",
            "print_step",
            "check_root",
            "check_requirements",
            "install_docker",
            "create_deployment_user",
            "setup_ssh_keys",
            "configure_firewall",
            "install_essential_tools",
            "create_app_directories",
            "setup_app_config",
            "deploy_application",
            "start_application",
            "create_backup",
            "rollback",
            "verify_deployment",
            "main"
        ]
        
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        for func in required_functions:
            # Check for function definition
            function_pattern = f"{func}()"
            self.assertIn(
                function_pattern, script_content,
                f"Required function '{func}' not found in script"
            )
    
    def test_error_handling_enabled(self):
        """Test that proper error handling is enabled."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Check for 'set -e' (exit on error)
        self.assertIn("set -e", script_content)
        
        # Check for 'set -o pipefail' (catch errors in pipes)
        self.assertIn("set -o pipefail", script_content)
    
    def test_configuration_variables_defined(self):
        """Test that required configuration variables are defined."""
        required_vars = [
            "DEPLOYMENT_USER",
            "APP_DIR",
            "BACKUP_DIR",
            "LOG_DIR",
            "CONFIG_FILE",
            "SSH_KEY_PATH",
            "DOCKER_VERSION",
            "REQUIRED_DISK_SPACE_GB"
        ]
        
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        for var in required_vars:
            self.assertIn(
                var, script_content,
                f"Required configuration variable '{var}' not found"
            )


class TestDeploymentScriptFunctionality(unittest.TestCase):
    """Test the functionality of deployment script components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_help_command(self):
        """Test that the help command works."""
        result = subprocess.run(
            ["bash", str(self.script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        # Help should exit with 0
        self.assertEqual(result.returncode, 0)
        
        # Help output should contain usage information
        self.assertIn("Usage:", result.stdout)
        self.assertIn("Commands:", result.stdout)
    
    def test_invalid_command(self):
        """Test that invalid commands are handled properly."""
        result = subprocess.run(
            ["bash", str(self.script_path), "invalid_command"],
            capture_output=True,
            text=True
        )
        
        # Invalid command should exit with non-zero
        self.assertNotEqual(result.returncode, 0)
        
        # Error message should be displayed
        self.assertIn("Unknown command", result.stdout + result.stderr)


class TestDeploymentScriptConfiguration(unittest.TestCase):
    """Test configuration handling in the deployment script."""
    
    def test_default_configuration_values(self):
        """Test that default configuration values are properly set."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Check default values
        default_configs = {
            "DEPLOYMENT_USER": "tradingbot",
            "APP_DIR": "/opt/trading-bot",
            "BACKUP_DIR": "/opt/trading-bot-backups",
            "LOG_DIR": "/var/log/trading-bot"
        }
        
        for var, default_value in default_configs.items():
            # Check if variable is defined with default value
            self.assertIn(f'{var}="${{{var}:-{default_value}}}"', script_content)


class TestDeploymentScriptSecurity(unittest.TestCase):
    """Test security aspects of the deployment script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
    
    def test_no_hardcoded_credentials(self):
        """Test that no credentials are hardcoded in the script."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Patterns that might indicate hardcoded credentials
        suspicious_patterns = [
            "password=",
            "api_key=",
            "secret=",
            "token="
        ]
        
        for pattern in suspicious_patterns:
            # Allow these patterns in comments or variable names
            lines = script_content.split('\n')
            for line in lines:
                if pattern.lower() in line.lower() and not line.strip().startswith('#'):
                    # Check if it's a variable assignment with template/placeholder
                    if "your_" not in line.lower() and "$" not in line:
                        self.fail(f"Potential hardcoded credential found: {line}")
    
    def test_ssh_key_authentication_configured(self):
        """Test that SSH key authentication is properly configured."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Should disable password authentication
        self.assertIn("PasswordAuthentication no", script_content)
        
        # Should enable public key authentication
        self.assertIn("PubkeyAuthentication yes", script_content)
    
    def test_firewall_configuration_exists(self):
        """Test that firewall configuration is implemented."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Should configure UFW
        self.assertIn("ufw", script_content)
        
        # Should allow SSH
        self.assertIn("ufw allow 22", script_content)
        
        # Should enable firewall
        self.assertIn("ufw --force enable", script_content)


class TestDeploymentScriptDocumentation(unittest.TestCase):
    """Test documentation and comments in the deployment script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
    
    def test_script_has_header_comments(self):
        """Test that the script has proper header documentation."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Should have description of what the script does
        self.assertIn("Vultr", script_content[:500])
        self.assertIn("deployment", script_content[:500].lower())
    
    def test_functions_have_comments(self):
        """Test that major functions have documentation comments."""
        with open(self.script_path, 'r') as f:
            lines = f.readlines()
        
        function_count = 0
        documented_functions = 0
        
        for i, line in enumerate(lines):
            if line.strip().endswith("() {"):
                function_count += 1
                # Check if previous lines contain comments
                for j in range(max(0, i-5), i):
                    if lines[j].strip().startswith('#'):
                        documented_functions += 1
                        break
        
        # At least 70% of functions should have documentation
        if function_count > 0:
            documentation_ratio = documented_functions / function_count
            self.assertGreater(
                documentation_ratio, 0.7,
                f"Only {documentation_ratio*100:.1f}% of functions are documented"
            )


class TestDeploymentConfigurationFiles(unittest.TestCase):
    """Test deployment configuration files."""
    
    def test_deploy_config_example_exists(self):
        """Test that deploy.config.example exists."""
        config_path = Path(__file__).parent.parent.parent / "config" / "deploy.config.example"
        self.assertTrue(config_path.exists())
    
    def test_env_production_template_exists(self):
        """Test that .env.production template exists."""
        env_path = Path(__file__).parent.parent.parent / ".env.production"
        self.assertTrue(env_path.exists())
    
    def test_production_env_has_required_vars(self):
        """Test that production environment template has required variables."""
        env_path = Path(__file__).parent.parent.parent / ".env.production"
        
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        required_vars = [
            "TRADING_ENV",
            "DEBUG",
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "LOG_LEVEL",
            "ENABLE_DISCORD_NOTIFICATIONS"
        ]
        
        for var in required_vars:
            self.assertIn(
                var, env_content,
                f"Required environment variable '{var}' not found in production template"
            )


class TestDeploymentScriptRobustness(unittest.TestCase):
    """Test robustness and error handling of the deployment script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
    
    def test_rollback_function_exists(self):
        """Test that rollback functionality is implemented."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        self.assertIn("rollback()", script_content)
        self.assertIn("trap", script_content)  # Should have error trap for rollback
    
    def test_backup_before_deployment(self):
        """Test that backup is created before deployment."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        # Should call create_backup before deployment
        self.assertIn("create_backup", script_content)
    
    def test_verification_after_deployment(self):
        """Test that deployment verification is performed."""
        with open(self.script_path, 'r') as f:
            script_content = f.read()
        
        self.assertIn("verify_deployment", script_content)


class TestDeploymentScriptIntegration(unittest.TestCase):
    """Integration tests for deployment script with existing project structure."""
    
    def test_compatible_with_docker_compose(self):
        """Test that deployment script references correct docker-compose files."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Should reference production docker-compose file
        self.assertIn("docker-compose.prod.yml", script_content)
    
    def test_compatible_with_project_structure(self):
        """Test that deployment script uses correct project directories."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "deploy.sh"
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Should create necessary directories
        required_dirs = ["logs", "data", "config"]
        for dir_name in required_dirs:
            self.assertIn(dir_name, script_content)


def run_deployment_tests():
    """Run all deployment script tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptDocumentation))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentConfigurationFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentScriptIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_deployment_tests()
    sys.exit(0 if success else 1)
