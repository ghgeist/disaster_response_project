# Gemini CLI Agent Interaction Guide

This document provides instructions on how to interact with the Signal Storm project using the Gemini CLI agent.

## Project Overview

Signal Storm is a machine learning pipeline designed to classify emergency messages into 36 disaster-related categories. This helps emergency response agencies prioritize and route assistance effectively during natural disasters.

## Getting Started with the CLI

The Gemini CLI agent can help you set up the environment, run the application, and manage the project.

### Environment Setup

To set up the development environment, you can ask the agent to perform the following steps:

1.  **Create a virtual environment:**
    ```
    create a virtual environment
    ```
2.  **Install dependencies:**
    ```
    install the dependencies from requirements.txt
    ```
3. **Install the local package:**
    ```
    install the local package in editable mode
    ```
4.  **Process the data:**
    ```
    run the data processing script
    ```

### Running the Web Application

To start the Flask web application, use the following command:

```
run the web application
```

The agent will execute `python run.py`, and the application will be available at `http://localhost:5000`.

## Interacting with the Project

Here are some examples of how you can use the Gemini CLI agent to interact with the project:

### Training a Model

You can ask the agent to train a new model. For example, to train a new production model:

```
train a new production model
```

To train a lightweight model:

```
train a new lightweight model
```

### Running Experiments

The project is set up to run experiments with different sampling strategies and hyperparameters. You can ask the agent to run these experiments for you.

*   **Test sampling strategies:**
    ```
    test the sampling strategies
    ```
*   **Test hyperparameters:**
    ```
    test the hyperparameters
    ```
*   **Compare model results:**
    ```
    compare the model results
    ```

### Exploring the Codebase

You can ask the agent to help you understand the project's structure and find specific code.

*   **List files in a directory:**
    ```
    list the files in the scripts directory
    ```
*   **Read a file:**
    ```
    read the contents of app/routes.py
    ```
*   **Find where a function is defined:**
    ```
    where is the create_app function defined?
    ```

### Running Tests

To run the project's tests, you can use the following commands:

*   **Run all tests:**
    ```
    run all tests
    ```
*   **Run specific tests:**
    ```
    run the tests in tests/test_app_smoke.py
    ```

This guide provides a starting point for interacting with the Signal Storm project using the Gemini CLI agent. You can use natural language to ask the agent to perform a wide variety of tasks related to this project.
