# Expert Reasoning System

A high-performance reasoning system that evaluates complex questions using multiple expert LLM agents with different strengths and specializations.

## Overview

This system implements a multi-expert approach to problem solving, leveraging the strengths of different large language models (LLMs) with specialized tools and strategic routing. The system:

1. Analyzes and classifies questions based on type and complexity
2. Routes questions to appropriate processing pipelines
3. Generates multiple expert responses using reasoning-optimized prompts
4. Executes code for computational problems when needed
5. Searches the web for knowledge-intensive questions
6. Evaluates all responses and selects the most accurate answer
7. Provides detailed analysis and metrics on system performance

## Features

- **Multiple Expert Models**: Leverages GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro, and other state-of-the-art models
- **Advanced Reasoning**: Uses specialized reasoning techniques like chain-of-thought and high reasoning effort
- **Code Generation & Execution**: Automatically generates and executes Python code for computational problems
- **Web Search Integration**: Uses Gemini's search capabilities for knowledge-intensive questions
- **Strategic Router**: Intelligently routes questions to the optimal processing pipeline
- **Comprehensive Evaluation**: Verifies responses against known answers and selects the best one
- **Detailed Analytics**: Provides performance metrics by question type, category, and expert

## Architecture

The system follows a clean, modular architecture:

- **Agent Layer**: Handles interactions with various LLM providers
- **Tools Layer**: Provides specialized capabilities like search and code execution
- **Memory Layer**: Manages storage and analysis of results
- **Routing Layer**: Determines the optimal strategy for each question

## Installation

1. Clone the repository
2. Install dependencies
3. Configure API keys
Create a `.env` file in the root directory with your API keys:

## Usage

Run the system on a dataset of questions:

python main.py --file test_hle.xlsx --sample 100