"""
User interaction utilities for the training interface.
"""


def get_user_input(prompt):
    """
    Get user input with validation.

    This function prompts the user for input and validates it.
    The function continues to prompt the user until they enter 'yes', 'no', or 'exit' (case insensitive).

    Args:
    prompt (str): The prompt to display to the user.

    Returns:
    user_input (str): The validated user input, converted to lowercase.
    """
    while True:
        user_input = input(prompt)
        if user_input.lower() in ["yes", "no", "exit"]:
            return user_input.lower()
        else:
            print("Invalid input. Please enter 'yes', 'no', or 'exit'.")
