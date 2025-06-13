"""
Utility Functions Module

This module provides various utility functions for working with LLM responses,
formatting messages, and other common tasks.
"""

import re
import json
import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def format_messages(role: str, content: str) -> Dict[str, str]:
    """
    Format a message for LLM API calls.
    
    Args:
        role: The role ('user', 'assistant', or 'system')
        content: The message content
        
    Returns:
        Formatted message dictionary
    """
    return {
        "role": role,
        "content": content
    }

def extract_json_from_string(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from a string that might contain other text.
    
    Args:
        text: String that might contain JSON
        
    Returns:
        Parsed JSON as dictionary, or empty dict if parsing fails
    """
    # Try to find JSON within the string (handles both raw JSON and markdown-formatted JSON)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find any JSON object in the text
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            logger.warning("No JSON found in string")
            return {}
    
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}

def clean_output(text: str) -> str:
    """
    Clean and format model output for presentation.
    
    Args:
        text: Raw text from model
        
    Returns:
        Cleaned and formatted text
    """
    # Remove code block markers if present
    text = re.sub(r'```(?:python|json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def parse_list_string(list_str: str) -> List[str]:
    """
    Parse a comma-separated string into a list.
    
    Args:
        list_str: Comma-separated string
        
    Returns:
        List of strings
    """
    if not list_str or list_str.lower() == "no":
        return []
    
    # Split by comma and clean each item
    items = [item.strip() for item in list_str.split(',')]
    return [item for item in items if item]

def extract_answer_from_response(response: str) -> str:
    """
    Extract the final answer from a detailed response.
    
    Args:
        response: The full response text
        
    Returns:
        Extracted answer
    """
    # Look for common answer indicators
    answer_patterns = [
        r"(?:final answer|answer)[:\s]+([^\n]+)",
        r"(?:conclusion|result)[:\s]+([^\n]+)",
        r"(?:therefore|thus|hence)[,\s]+([^\n]+)",
        r"(?:the answer is)[:\s]+([^\n]+)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no clear answer indicator found, return the last non-empty line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return ""

def calculate_response_metrics(response: str) -> Dict[str, Any]:
    """
    Calculate metrics about a response (length, complexity, etc.).
    
    Args:
        response: The response text
        
    Returns:
        Dictionary of metrics
    """
    # Count words
    words = re.findall(r'\b\w+\b', response)
    word_count = len(words)
    
    # Count sentences
    sentences = re.split(r'[.!?]+', response)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Count paragraphs
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_length": avg_sentence_length,
        "total_length": len(response)
    }