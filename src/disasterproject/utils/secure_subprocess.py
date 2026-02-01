import os
import re
import subprocess
import sys


class SecureSubprocessError(Exception):
    """Custom exception for security-related errors in subprocesses."""
    pass

def validate_file_path(file_path, must_exist=True, allow_absolute=False):
    """
    Validates a file path to prevent security risks like path traversal.

    Args:
        file_path (str): The file path to validate.
        must_exist (bool): If True, the file must exist.
        allow_absolute (bool): If True, allow absolute paths (use with caution).

    Returns:
        str: The validated file path.

    Raises:
        SecureSubprocessError: If the path is invalid or doesn't exist when required.
    """
    # Always block path traversal attempts
    if ".." in file_path:
        raise SecureSubprocessError("Path traversal detected")

    # Block absolute paths unless explicitly allowed
    if os.path.isabs(file_path) and not allow_absolute:
        raise SecureSubprocessError("Absolute paths not allowed")

    if must_exist and not os.path.exists(file_path):
        raise SecureSubprocessError("File does not exist")

    return file_path

def validate_command_args(args):
    """
    Validates command arguments to prevent command injection.

    Args:
        args (list): A list of command arguments.

    Returns:
        list: The validated list of arguments.

    Raises:
        SecureSubprocessError: If the arguments are not a list of strings or contain unsafe characters.
    """
    if not isinstance(args, list):
        raise SecureSubprocessError("Command arguments must be a list")

    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            raise SecureSubprocessError("All arguments must be strings")

        # Skip validation for sys.executable (first argument) to allow Windows paths with parentheses
        if i == 0 and arg == sys.executable:
            continue

        # Simplified character check - block shell metacharacters including parentheses
        for char in "&|;$`()<>\n":
            if char in arg:
                raise SecureSubprocessError("Unsafe characters detected in arguments")

    return args

def secure_run(command, timeout=None, shell=False):
    """
    A wrapper around subprocess.run that enforces security best practices.

    Args:
        command (list): The command to execute as a list of arguments.
        timeout (int, optional): The timeout for the command. Defaults to None.
        shell (bool): This is ignored and always set to False for security.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess execution.
    """
    validated_command = validate_command_args(command)
    return subprocess.run(validated_command, timeout=timeout, shell=False, check=True, capture_output=True, text=True)

def secure_python_script(script_path, args=None):
    """
    Securely executes a Python script.

    Args:
        script_path (str): The path to the Python script to execute.
        args (list, optional): A list of arguments for the script. Defaults to None.

    Returns:
        subprocess.CompletedProcess: The result of the script execution.
    """
    validate_file_path(script_path)
    if not script_path.endswith(".py"):
        raise SecureSubprocessError("Script must be a Python file")

    command = [sys.executable, script_path]
    if args:
        command.extend(validate_command_args(args))

    return secure_run(command)

def validate_model_filename(filename):
    """
    Validates a model filename to ensure it is safe.

    Args:
        filename (str): The filename to validate.

    Returns:
        str: The validated filename, with .pkl extension.

    Raises:
        SecureSubprocessError: If the filename contains invalid characters.
    """
    # Simplified regex
    if not re.match(r'^[a-zA-Z0-9_\-.]+$', filename):
        raise SecureSubprocessError("Invalid characters in filename")

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    return filename

def validate_sampling_method(method):
    """
    Validates the sampling method to prevent injection attacks.

    Args:
        method (str): The sampling method to validate.

    Returns:
        str: The validated sampling method in lowercase.

    Raises:
        SecureSubprocessError: If the method is unknown or contains invalid characters.
    """
    allowed_methods = ["baseline", "smote", "adasyn", "conservative", "random", "borderline"]
    # Simplified regex
    if not re.match(r'^[a-zA-Z_]+$', method):
        raise SecureSubprocessError("Invalid characters in method name")

    if method.lower() not in allowed_methods:
        raise SecureSubprocessError("Unknown sampling method")

    return method.lower()
