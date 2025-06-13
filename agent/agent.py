"""
LLM Agent Implementation Module

This module implements various LLM agents for different providers (OpenAI, Anthropic, Google, etc.)
Each agent provides a consistent interface for calling different LLM models.
"""

import os
import httpx
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from openai import OpenAI, AsyncOpenAI, DefaultHttpxClient
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types

# Initialize logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BaseAgent(ABC):
    """Abstract base class for LLM agents."""
    
    @abstractmethod
    async def generate_response(self, 
                               messages: List[Dict[str, str]], 
                               model_name: Optional[str] = None, 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_reasoning_response(self, 
                                         messages: List[Dict[str, str]], 
                                         model_name: Optional[str] = None, 
                                         system_message: Optional[str] = None,
                                         temperature: float = 0) -> str:
        """Generate a response with explicit reasoning from the LLM."""
        pass

class GPTAgent(BaseAgent):
    """Agent for OpenAI's models (GPT-4, etc.)."""
    
    def __init__(self):
        """Initialize the GPT agent with configured clients."""
        # Standard client for synchronous calls
        self.client = OpenAI(
            http_client=DefaultHttpxClient(
                proxy=os.getenv("PROXY_URL"),
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            ),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Async client for asynchronous calls
        self.async_client = AsyncOpenAI(
            http_client=httpx.AsyncClient(
                proxies=os.getenv("PROXY_URL"),
                timeout=180.0,
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            ),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Special client for reasoning tasks
        self.async_reason_client = AsyncOpenAI(
            http_client=httpx.AsyncClient(
                proxies=os.getenv("PROXY_URL"),
                timeout=180.0,
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            ),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info("GPT Agent initialized successfully")
    
    async def generate_response(self, 
                               messages: List[Dict[str, str]], 
                               model_name: str = "gpt-4o", 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """
        Generate a response using OpenAI's models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The OpenAI model to use
            system_message: Optional system message to prepend
            temperature: Sampling temperature (0-2)
            
        Returns:
            Generated text response
        """
        if system_message:
            sys_msg = {
                "role": "system",
                "content": system_message
            }
            messages.insert(0, sys_msg)
            
        try:
            result = await self.async_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stream=False
            )
            
            return result.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling GPT model {model_name}: {str(e)}")
            return ""
    
    async def generate_reasoning_response(self, 
                                         messages: List[Dict[str, str]], 
                                         model_name: str = "o3-mini", 
                                         system_message: Optional[str] = None,
                                         temperature: float = 0) -> str:
        """
        Generate a response with explicit reasoning from OpenAI models.
        
        This method uses models that support the reasoning_effort parameter
        or other similar reasoning capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The OpenAI model to use (should support reasoning)
            system_message: Optional system message to prepend
            temperature: Sampling temperature (0-2)
            
        Returns:
            Generated text response with reasoning
        """
        if system_message:
            sys_msg = {
                "role": "system",
                "content": system_message
            }
            messages.insert(0, sys_msg)
            
        try:
            result = await self.async_reason_client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "text"},
                reasoning_effort="high"
            )
            
            return result.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling GPT reasoning model {model_name}: {str(e)}")
            return ""


class ClaudeAgent(BaseAgent):
    """Agent for Anthropic's Claude models."""
    
    def __init__(self):
        """Initialize the Claude agent with configured clients."""
        # Standard client for synchronous calls
        self.client = Anthropic(
            http_client=httpx.Client(
                proxies=os.getenv("PROXY_URL"),
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            ),
            api_key=os.getenv("CLAUDE_API_KEY")
        )
        
        # Async client for asynchronous calls
        self.async_client = AsyncAnthropic(
            http_client=httpx.AsyncClient(
                proxies=os.getenv("PROXY_URL"),
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            ),
            api_key=os.getenv("CLAUDE_API_KEY")
        )
        
        logger.info("Claude Agent initialized successfully")
    
    async def generate_response(self, 
                               messages: List[Dict[str, str]], 
                               model_name: str = "claude-3-5-sonnet-20241022", 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """
        Generate a response using Claude models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The Claude model to use
            system_message: Optional system message
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        try:
            if system_message:
                response = await self.async_client.with_options(timeout=120.0).messages.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=4096,
                    system=system_message,
                    messages=messages
                )
            else:
                response = await self.async_client.with_options(timeout=120.0).messages.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=4096,
                    messages=messages
                )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude model {model_name}: {str(e)}")
            return ""
    
    async def generate_reasoning_response(self, 
                                         messages: List[Dict[str, str]], 
                                         model_name: str = "claude-3-5-sonnet-20241022", 
                                         system_message: Optional[str] = None,
                                         temperature: float = 1) -> str:
        """
        Generate a response with explicit reasoning from Claude models.
        
        This method uses Claude's "thinking" parameter to generate detailed reasoning.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The Claude model to use
            system_message: Optional system message
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response with reasoning
        """
        try:
            if system_message:
                response = await self.async_client.with_options(timeout=120.0).messages.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=8192,
                    system=system_message,
                    messages=messages,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 4096
                    }
                )
            else:
                response = await self.async_client.with_options(timeout=120.0).messages.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=8192,
                    messages=messages,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 4096
                    }
                )
            return response.content[1].text
        except Exception as e:
            logger.error(f"Error calling Claude reasoning model {model_name}: {str(e)}")
            return ""


class GeminiAgent(BaseAgent):
    """Agent for Google's Gemini models."""
    
    def __init__(self):
        """Initialize the Gemini agent with configured client."""
        # Initialize the Gemini client
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={'api_version': 'v1alpha'},
        )
        
        logger.info("Gemini Agent initialized successfully")
    
    async def generate_response(self, 
                               messages: Union[List[Dict[str, str]], str], 
                               model_name: str = "gemini-2.0-flash-exp", 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """
        Generate a response using Gemini models.
        
        Args:
            messages: Either a string or list of message dictionaries 
            model_name: The Gemini model to use
            system_message: Optional system message (not used for Gemini)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        try:
            result = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[messages],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            
            return result.text
        except Exception as e:
            logger.error(f"Error calling Gemini model {model_name}: {str(e)}")
            return ""
    
    async def generate_reasoning_response(self, 
                                         messages: Union[List[Dict[str, str]], str], 
                                         model_name: str = "gemini-2.0-flash-exp", 
                                         system_message: Optional[str] = None,
                                         temperature: float = 0) -> str:
        """
        Generate a response with reasoning from Gemini models.
        
        For Gemini, this currently uses the same method as generate_response,
        but could be enhanced if Gemini provides specific reasoning parameters.
        
        Args:
            messages: Either a string or list of message dictionaries
            model_name: The Gemini model to use
            system_message: Optional system message (not used for Gemini)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        # For now, this uses the same method as generate_response
        # Could be enhanced if Gemini provides specific reasoning parameters
        return await self.generate_response(messages, model_name, system_message, temperature)
    
    async def search_and_generate(self, 
                                 query: str, 
                                 model_name: str = "gemini-2.5-pro-preview-03-25",
                                 temperature: float = 0) -> str:
        """
        Perform web search and generate content using Gemini models.
        
        This method uses Gemini's built-in Google Search capability.
        
        Args:
            query: The search query or user question
            model_name: The Gemini model to use (should support search)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated response that incorporates search results
        """
        try:
            tools = [
                types.Tool(google_search=types.GoogleSearch())
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                tools=tools
            )
            result = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[query],
                config=generate_content_config
            )
            
            return result.text
        except Exception as e:
            logger.error(f"Error calling Gemini search model {model_name}: {str(e)}")
            return ""
    
    async def code_and_generate(self, 
                                 query: str, 
                                 model_name: str = "gemini-2.5-pro-preview-03-25",
                                 temperature: float = 0) -> str:
        """
        Perform code execution and generate content using Gemini models.
        
        This method uses Gemini's built-in Google Code capability.
        
        Args:
            query: The search query or user question
            model_name: The Gemini model to use (should support search)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated response that incorporates search results
        """
        try:
            tools = [
                types.Tool(code_execution=types.ToolCodeExecution),
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                tools=tools
            )
            result = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[query],
                config=generate_content_config
            )
            
            return result.text
        except Exception as e:
            logger.error(f"Error calling Gemini code model {model_name}: {str(e)}")
            return ""


class GeminiEvaluatorAgent(BaseAgent):
    """Agent for Google's Gemini models."""
    
    def __init__(self):
        """Initialize the Gemini agent with configured client."""
        # Initialize the Gemini client
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY_EVALUTOR"),
            http_options={'api_version': 'v1alpha'},
        )
        
        logger.info("Gemini Agent initialized successfully")
    
    async def generate_response(self, 
                               messages: Union[List[Dict[str, str]], str], 
                               model_name: str = "gemini-2.0-flash-exp", 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """
        Generate a response using Gemini models.
        
        Args:
            messages: Either a string or list of message dictionaries 
            model_name: The Gemini model to use
            system_message: Optional system message (not used for Gemini)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        try:
            result = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[messages],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            
            return result.text
        except Exception as e:
            logger.error(f"Error calling Gemini model {model_name}: {str(e)}")
            return ""
    
    async def generate_reasoning_response(self, 
                                         messages: Union[List[Dict[str, str]], str], 
                                         model_name: str = "gemini-2.0-flash-exp", 
                                         system_message: Optional[str] = None,
                                         temperature: float = 0) -> str:
        """
        Generate a response with reasoning from Gemini models.
        
        For Gemini, this currently uses the same method as generate_response,
        but could be enhanced if Gemini provides specific reasoning parameters.
        
        Args:
            messages: Either a string or list of message dictionaries
            model_name: The Gemini model to use
            system_message: Optional system message (not used for Gemini)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        # For now, this uses the same method as generate_response
        # Could be enhanced if Gemini provides specific reasoning parameters
        return await self.generate_response(messages, model_name, system_message, temperature)
    
    async def search_and_generate(self, 
                                 query: str, 
                                 model_name: str = "gemini-2.5-pro-preview-03-25",
                                 temperature: float = 0) -> str:
        """
        Perform web search and generate content using Gemini models.
        
        This method uses Gemini's built-in Google Search capability.
        
        Args:
            query: The search query or user question
            model_name: The Gemini model to use (should support search)
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated response that incorporates search results
        """
        try:
            tools = [
                types.Tool(google_search=types.GoogleSearch())
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                tools=tools
            )
            result = await self.client.aio.models.generate_content(
                model=model_name,
                contents=[query],
                config=generate_content_config
            )
            
            return result.text
        except Exception as e:
            logger.error(f"Error calling Gemini Evaluator search model {model_name}: {str(e)}")
            return ""


class TongyiAgent(BaseAgent):
    """Agent for Alibaba's Tongyi models."""
    
    def __init__(self):
        """Initialize the Tongyi agent with configured clients."""
        # Standard client for synchronous calls
        self.client = OpenAI(
            api_key=os.getenv("ALI_API_KEY"),
            base_url=os.getenv("ALI_BASE_URL"),
        )
        
        # Async client for asynchronous calls
        self.async_client = AsyncOpenAI(
            api_key=os.getenv("ALI_API_KEY"),
            base_url=os.getenv("ALI_BASE_URL"),
            http_client=httpx.AsyncClient(
                proxies=os.getenv("PROXY_URL"),
                timeout=180.0,
                transport=httpx.HTTPTransport(local_address=os.getenv("LOCAL_ADDRESS")),
            )
        )
        
        logger.info("Tongyi Agent initialized successfully")
    
    async def generate_response(self, 
                               messages: List[Dict[str, str]], 
                               model_name: str = "qwen-plus", 
                               system_message: Optional[str] = None,
                               temperature: float = 0) -> str:
        """
        Generate a response using Tongyi models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The Tongyi model to use
            system_message: Optional system message to prepend
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        if system_message:
            sys_msg = {
                "role": "system",
                "content": system_message
            }
            messages.insert(0, sys_msg)
            
        try:
            result = await self.async_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            
            return result.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Tongyi model {model_name}: {str(e)}")
            return ""
    
    async def generate_reasoning_response(self, 
                                         messages: List[Dict[str, str]], 
                                         model_name: str = "qwen-plus", 
                                         system_message: Optional[str] = None,
                                         temperature: float = 0) -> str:
        """
        Generate a response with reasoning from Tongyi models.
        
        For Tongyi, this currently uses the same method as generate_response,
        but with a system message that requests reasoning.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: The Tongyi model to use
            system_message: Optional system message to prepend
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response with reasoning
        """
        if not system_message:
            system_message = "Please provide detailed reasoning for your response, explaining your thought process step by step."
            
        return await self.generate_response(messages, model_name, system_message, temperature)