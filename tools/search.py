"""
Web Search Tool Module

This module provides tools for searching the web and retrieving information
using various search APIs and information retrieval techniques.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from agent.agent import GeminiAgent


logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Tool for performing web searches using available APIs.
    """
    
    def __init__(self):
        """Initialize the web search tool."""
        self.gemini_agent = GeminiAgent()
        logger.info("Web Search Tool initialized")
    
    async def search(self, query: str, temperature: float = 1.0) -> str:
        """
        Perform a web search using Gemini's search capability and return results.
        
        Args:
            query: The search query or user question
            temperature: Sampling temperature for generation
            
        Returns:
            Generated response that incorporates search results
        """
        # Format the query for better search results
        search_prompt = f"""
            <User Question>
            {query}
            </User Question>

            Please answer the question above based on the following principles:
            - Carefully read and analyze the information retrieved. Search in English and answer in English.
            - Ensure that the search keywords are closely aligned with the user's question.
            - Think thoroughly step by step, and derive the answer step by step.
            - Be cautious during the reasoning process to avoid any traps or misleading setups in the question.
            - Review and verify the answer once more.
            - Finally, output the confirmed and final answer conclusion.
        """
        
        logger.info(f"Performing web search for query: {query}")
        result = await self.gemini_agent.search_and_generate(search_prompt, "gemini-2.5-pro-preview-03-25", temperature)
        return result
    
    async def multi_search(self, query: str, num_searches: int = 3) -> List[str]:
        """
        Perform multiple searches with different parameters and return all results.
        
        Args:
            query: The search query or user question
            num_searches: Number of parallel searches to perform
            
        Returns:
            List of search results
        """
        # Create tasks for multiple searches with different temperatures
        tasks = [
            self.search(query, temperature=0.7 + i * 0.2) 
            for i in range(num_searches)
        ]
        
        # Run searches in parallel
        results = await asyncio.gather(*tasks)
        return results