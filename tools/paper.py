"""
Paper Analysis Tool Module

This module provides tools for analyzing academic papers, extracting information,
and generating insights from research publications.
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PaperAnalysisTool:
    """
    Tool for analyzing academic papers and extracting information.
    """
    
    def __init__(self):
        """Initialize the paper analysis tool."""
        logger.info("Paper Analysis Tool initialized")
    
    def extract_paper_sections(self, paper_text: str) -> Dict[str, str]:
        """
        Extract main sections from a paper's text content.
        
        Args:
            paper_text: The full text content of the paper
            
        Returns:
            Dictionary mapping section names to their content
        """
        # Common section patterns in academic papers
        section_patterns = [
            r"(?i)abstract",
            r"(?i)introduction",
            r"(?i)related work",
            r"(?i)background",
            r"(?i)methodology|methods",
            r"(?i)experiments|experimental setup",
            r"(?i)results",
            r"(?i)discussion",
            r"(?i)conclusion",
            r"(?i)references"
        ]
        
        sections = {}
        
        # Find each section and its content
        for i, pattern in enumerate(section_patterns):
            matches = re.finditer(pattern, paper_text, re.IGNORECASE)
            
            for match in matches:
                section_start = match.start()
                section_name = paper_text[section_start:section_start + len(match.group(0))]
                
                # Find the next section start or end of document
                next_section_start = len(paper_text)
                for next_pattern in section_patterns[i+1:]:
                    next_match = re.search(next_pattern, paper_text[section_start:], re.IGNORECASE)
                    if next_match:
                        next_section_start = min(next_section_start, section_start + next_match.start())
                
                # Extract section content
                section_content = paper_text[section_start:next_section_start].strip()
                sections[section_name] = section_content
        
        return sections
    
    def extract_key_findings(self, paper_text: str) -> List[str]:
        """
        Extract key findings from a paper.
        
        Args:
            paper_text: The full text content of the paper
            
        Returns:
            List of extracted key findings
        """
        # Find results and conclusion sections
        results_pattern = r"(?i)results.*?(?=(conclusion|\Z))"
        conclusion_pattern = r"(?i)conclusion.*?(?=(\Z|references))"
        
        results_match = re.search(results_pattern, paper_text, re.DOTALL)
        conclusion_match = re.search(conclusion_pattern, paper_text, re.DOTALL)
        
        findings = []
        
        # Extract findings from results section
        if results_match:
            results_text = results_match.group(0)
            # Look for statements with findings indicators
            finding_indicators = [
                "show", "found", "reveal", "demonstrate", "indicate", 
                "suggest", "significant", "improvement", "increase", "decrease"
            ]
            
            for indicator in finding_indicators:
                finding_pattern = f"[^.]*{indicator}[^.]*\\."
                matches = re.finditer(finding_pattern, results_text, re.IGNORECASE)
                for match in matches:
                    findings.append(match.group(0).strip())
        
        # Extract findings from conclusion section
        if conclusion_match:
            conclusion_text = conclusion_match.group(0)
            sentences = re.split(r'\.', conclusion_text)
            for sentence in sentences:
                # Filter for likely conclusion statements
                if any(word in sentence.lower() for word in ["conclud", "demonstrat", "show", "contribut", "future"]):
                    if sentence.strip():
                        findings.append(sentence.strip() + ".")
        
        return findings
    
    def extract_references(self, paper_text: str) -> List[Dict[str, str]]:
        """
        Extract references from a paper.
        
        Args:
            paper_text: The full text content of the paper
            
        Returns:
            List of extracted references with metadata
        """
        # Find references section
        references_pattern = r"(?i)references.*?(\Z)"
        references_match = re.search(references_pattern, paper_text, re.DOTALL)
        
        references = []
        
        if references_match:
            references_text = references_match.group(0)
            
            # Common reference formats
            # Example: [1] Author, A. (2020). Title. Journal, Volume(Issue), Pages.
            numbered_ref_pattern = r"\[\d+\].*?(?=(\[\d+\]|\Z))"
            
            # Example: Author, A. (2020). Title. Journal, Volume(Issue), Pages.
            author_year_pattern = r"[A-Z][a-z]+,\s[A-Z]\.\s\(\d{4}\).*?(?=([A-Z][a-z]+,\s[A-Z]\.\s\(\d{4}\)|\Z))"
            
            # Try numbered references first
            numbered_matches = re.finditer(numbered_ref_pattern, references_text, re.DOTALL)
            for match in numbered_matches:
                ref_text = match.group(0).strip()
                if ref_text:
                    # Extract reference number
                    ref_num_match = re.search(r"\[(\d+)\]", ref_text)
                    ref_num = ref_num_match.group(1) if ref_num_match else ""
                    
                    # Extract authors
                    authors_match = re.search(r"\]\s*(.*?)(?=\(\d{4}\))", ref_text)
                    authors = authors_match.group(1).strip() if authors_match else ""
                    
                    # Extract year
                    year_match = re.search(r"\((\d{4})\)", ref_text)
                    year = year_match.group(1) if year_match else ""
                    
                    # Extract title
                    title_match = re.search(r"\)\.?\s*(.*?)(?=\.\s*[A-Z])", ref_text)
                    title = title_match.group(1).strip() if title_match else ""
                    
                    references.append({
                        "number": ref_num,
                        "authors": authors,
                        "year": year,
                        "title": title,
                        "full_citation": ref_text
                    })
            
            # If no numbered references, try author-year format
            if not references:
                author_year_matches = re.finditer(author_year_pattern, references_text, re.DOTALL)
                for match in author_year_matches:
                    ref_text = match.group(0).strip()
                    if ref_text:
                        # Extract authors
                        authors_match = re.search(r"(.*?)(?=\(\d{4}\))", ref_text)
                        authors = authors_match.group(1).strip() if authors_match else ""
                        
                        # Extract year
                        year_match = re.search(r"\((\d{4})\)", ref_text)
                        year = year_match.group(1) if year_match else ""
                        
                        # Extract title
                        title_match = re.search(r"\)\.?\s*(.*?)(?=\.\s*[A-Z])", ref_text)
                        title = title_match.group(1).strip() if title_match else ""
                        
                        references.append({
                            "authors": authors,
                            "year": year,
                            "title": title,
                            "full_citation": ref_text
                        })
        
        return references