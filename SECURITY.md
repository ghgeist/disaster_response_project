# Security Guidelines

## Command Injection Prevention

This repository contains Python scripts that use `subprocess.run()` to execute external commands. To prevent command injection vulnerabilities, follow these guidelines:

### ✅ Safe Practices

1. **Use list arguments instead of shell strings**:
   ```python
   # ✅ GOOD - Safe
   subprocess.run([sys.executable, "script.py", "arg1", "arg2"])
   
   # ❌ BAD - Vulnerable to injection
   subprocess.run(f"python script.py {user_input}", shell=True)
   ```

2. **Validate and sanitize user inputs**:
   ```python
   # ✅ GOOD - Validate inputs
   if not all(c.isalnum() or c in '._-' for c in filename):
       raise ValueError("Invalid filename")
   
   # ❌ BAD - No validation
   subprocess.run([sys.executable, f"script.py", user_filename])
   ```

3. **Use `os.path.basename()` to prevent path traversal**:
   ```python
   # ✅ GOOD - Prevent path traversal
   safe_filename = os.path.basename(user_input)
   
   # ❌ BAD - Allows path traversal
   subprocess.run([sys.executable, f"script.py", user_input])
   ```

### 🔒 Current Security Measures

The following files have been secured against command injection:

- `run_all_experiments.py` - Validates model names and sampling methods
- `scripts/systematic_testing_framework.py` - Validates method names and filenames

### 🚨 Security Considerations

Since this is a public repository with internal scripts:

- **Risk Level**: Low (internal scripts, not client-facing)
- **Main Threat**: Accidental command injection during development
- **Protection**: Basic input validation and sanitization
- **No Need For**: Complex security frameworks or extensive validation

### 📝 For Future Development

When adding new subprocess calls:

1. Always use list arguments: `[command, arg1, arg2]`
2. Never use `shell=True` unless absolutely necessary
3. Validate any user-controlled inputs
4. Use `os.path.basename()` for file paths
5. Test with malicious inputs to verify protection

### 🧪 Testing Security

To test command injection protection:

```python
# Test with malicious input
malicious_input = "../../etc/passwd; rm -rf /"
try:
    # Your subprocess call here
    pass
except ValueError:
    print("✅ Security validation working")
```
