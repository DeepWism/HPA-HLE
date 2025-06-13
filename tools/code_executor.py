"""
Code Executor Tool Module

This module provides tools for generating, executing, and analyzing code
to solve computational problems.
"""

import logging
import subprocess
import tempfile
import os
import time
import asyncio
from typing import Dict, Any, Optional, Tuple

from agent.agent import ClaudeAgent

logger = logging.getLogger(__name__)

class CodeExecutor:
    """
    Tool for generating and executing code to solve problems.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the code executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.claude_agent = ClaudeAgent()
        logger.info(f"Code Executor initialized with timeout: {timeout}s")
    
    async def generate_code(self, problem: str) -> str:
        """
        Generate Python code to solve the given problem.
        
        Args:
            problem: The problem description
            
        Returns:
            Generated Python code
        """
        program_message = [{
            "role": "user",
            "content": f"""
                <用户问题>
                {problem}                              
                </用户问题> 

                请根据用户的问题编写python代码来解题。

                注意：
                - 涉及数学计算、物理计算、化学计算的问题请编写python代码，并且编写正确的python代码。
                - 编写代码时注意题目的约束条件。
                - 编写代码时注意不要遗漏import的相关包。
                - 输出python代码请直接输出代码，不要有其他内容。
                - 代码中需要print出结果值。
            """
        }]
        
        logger.info(f"Generating code for problem: {problem[:100]}...")
        code = await self.claude_agent.generate_response(
            program_message, 
            model_name="claude-3-7-sonnet-20250219",
            temperature=1
        )
        
        # Clean up the code by removing markdown code blocks if present
        code = code.replace("```python", "").replace("```", "").strip()
        return code
    
    async def execute_code(self, code: str, problem_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Execute the given Python code and return the results.
        
        Args:
            code: The Python code to execute
            problem_id: Optional identifier for the problem
            
        Returns:
            Tuple of (execution output, success flag)
        """
        if not code.strip():
            logger.warning("Empty code provided for execution")
            return "No code to execute", False
        
        # Create a temporary file for the code
        file_id = problem_id if problem_id else str(int(time.time()))
        temp_script_path = f"temp_script_{file_id}.py"
        
        try:
            # Write code to temporary file
            with open(temp_script_path, 'w', encoding="utf-8") as f:
                f.write(code)
            
            logger.info(f"Executing code in {temp_script_path}")
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_script_path], 
                capture_output=True, 
                text=True, 
                timeout=self.timeout, 
                encoding='utf-8', 
                errors='replace'
            )
            
            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                logger.error(f"Code execution failed: {error_msg}")
                return f"Execution Error: {error_msg}", False
            
            output = result.stdout.strip()
            logger.info(f"Code executed successfully, output length: {len(output)}")
            return output, True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Code execution timed out after {self.timeout} seconds")
            return f"Execution timed out after {self.timeout} seconds", False
            
        except Exception as e:
            logger.error(f"Error during code execution: {str(e)}")
            return f"Execution error: {str(e)}", False
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    async def solve_with_code(self, problem: str, problem_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a problem by generating and executing code.
        
        Args:
            problem: The problem description
            problem_id: Optional identifier for the problem
            
        Returns:
            Dictionary containing the code, execution result, and status
        """
        logger.info(f"Solving problem with code: {problem_id or 'unknown'}")
        
        # Generate code
        code = await self.generate_code(problem)
        
        # Execute code
        output, success = await self.execute_code(code, problem_id)
        
        return {
            "code": code,
            "execution_output": output,
            "success": success,
            "problem_id": problem_id
        }