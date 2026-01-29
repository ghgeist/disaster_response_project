"""
Shared Flask extensions for the Disaster Response application.
"""
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
