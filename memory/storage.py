"""
Result Storage Module

This module provides functionality for storing, retrieving, and analyzing
results from reasoning experiments.
"""

import os
import json
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultStorage:
    """
    Storage and analysis of reasoning experiment results.
    """
    
    def __init__(self, storage_dir: str = "results"):
        """
        Initialize the result storage.
        
        Args:
            storage_dir: Directory for storing results
        """
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        logger.info(f"Result Storage initialized with directory: {storage_dir}")
    
    def save_results(self, 
                    results: List[List[Any]], 
                    columns: List[str], 
                    filename: Optional[str] = None) -> str:
        """
        Save results to an Excel file.
        
        Args:
            results: List of result rows
            columns: Column names for the results
            filename: Optional filename (generated if not provided)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.xlsx"
        
        # Ensure filename has .xlsx extension
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        # Create full path
        file_path = os.path.join(self.storage_dir, filename)
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(results, columns=columns)
        df.to_excel(file_path, index=False)
        
        logger.info(f"Results saved to: {file_path}")
        return file_path
    
    def save_detailed_result(self, 
                            result: Dict[str, Any], 
                            question_id: Union[str, int]) -> str:
        """
        Save a detailed result for a single question in JSON format.
        
        Args:
            result: Dictionary containing detailed results
            question_id: Identifier for the question
            
        Returns:
            Path to the saved file
        """
        # Create directory for detailed results if it doesn't exist
        detailed_dir = os.path.join(self.storage_dir, "detailed")
        if not os.path.exists(detailed_dir):
            os.makedirs(detailed_dir)
        
        # Generate filename
        filename = f"question_{question_id}.json"
        file_path = os.path.join(detailed_dir, filename)
        
        # Add timestamp to result
        result['timestamp'] = datetime.now().isoformat()
        
        # Save to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed result for question {question_id} saved to: {file_path}")
        return file_path
    
    def load_results(self, filename: str) -> pd.DataFrame:
        """
        Load results from an Excel file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame containing the results
        """
        file_path = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Results file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_excel(file_path)
        logger.info(f"Loaded results from: {file_path}")
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze results to extract insights and metrics.
        
        Args:
            results_df: DataFrame containing results
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Calculate overall accuracy
        if "is_correct" in results_df.columns:
            correct_count = (results_df["is_correct"] == "yes").sum()
            total_count = len(results_df)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            analysis["accuracy"] = accuracy
            analysis["correct_count"] = correct_count
            analysis["total_count"] = total_count
        
        # Calculate accuracy by question type
        if "answer_type" in results_df.columns and "is_correct" in results_df.columns:
            type_analysis = {}
            for answer_type in results_df["answer_type"].unique():
                type_df = results_df[results_df["answer_type"] == answer_type]
                type_correct = (type_df["is_correct"] == "yes").sum()
                type_total = len(type_df)
                type_accuracy = type_correct / type_total if type_total > 0 else 0
                
                type_analysis[answer_type] = {
                    "accuracy": type_accuracy,
                    "correct_count": type_correct,
                    "total_count": type_total
                }
            
            analysis["by_answer_type"] = type_analysis
        
        # Calculate accuracy by category
        if "category" in results_df.columns and "is_correct" in results_df.columns:
            category_analysis = {}
            for category in results_df["category"].unique():
                cat_df = results_df[results_df["category"] == category]
                cat_correct = (cat_df["is_correct"] == "yes").sum()
                cat_total = len(cat_df)
                cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
                
                category_analysis[category] = {
                    "accuracy": cat_accuracy,
                    "correct_count": cat_correct,
                    "total_count": cat_total
                }
            
            analysis["by_category"] = category_analysis
        
        logger.info("Results analysis completed")
        return analysis
    
    def export_analysis(self, analysis: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export analysis results to a JSON file.
        
        Args:
            analysis: Dictionary of analysis results
            filename: Optional filename (generated if not provided)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
        
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create full path
        file_path = os.path.join(self.storage_dir, filename)
        
        # Add timestamp to analysis
        analysis['timestamp'] = datetime.now().isoformat()

        # jsonEncoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Save to JSON 
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Analysis saved to: {file_path}")
        return file_path