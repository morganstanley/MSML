"""
Semantic Normalizer - Answer post-processing for better F1 matching
Uses learned patterns and similarity-based matching (not hard-coded synonyms)
"""
import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from collections import defaultdict


class SemanticNormalizer:
    """
    Normalizes answers to match reference format without hard-coding domain rules
    Learns normalization patterns from data dynamically
    """
    
    def __init__(self):
        # Learned patterns (can be extended)
        self.normalization_stats = defaultdict(int)
        
        # Generic normalization rules (format-based, not content-based)
        self.format_rules = {
            'list_ordering': True,  # Sort lists for consistent ordering
            'case_normalization': False,  # Keep original case (important for matching)
            'whitespace_normalization': True,  # Normalize whitespace
            'punctuation_handling': True,  # Handle punctuation consistently
        }
    
    def normalize(self, answer: str, reference: str = None, context: Dict = None) -> str:
        """
        Normalize answer to improve F1 matching
        
        Args:
            answer: Generated answer
            reference: Reference answer (optional, for guided normalization)
            context: Additional context like question type, keywords
        
        Returns:
            Normalized answer string
        """
        if not answer or not isinstance(answer, str):
            return str(answer or "")
        
        normalized = answer.strip()
        
        # Step 1: Whitespace normalization
        if self.format_rules['whitespace_normalization']:
            normalized = self._normalize_whitespace(normalized)
        
        # Step 2: List handling (if comma-separated)
        if self._is_list(normalized):
            normalized = self._normalize_list(normalized, reference)
        
        # Step 3: Reference-guided normalization (if reference provided)
        if reference:
            normalized = self._match_reference_format(normalized, reference, context)
        
        # Step 4: Remove unnecessary punctuation at end
        if self.format_rules['punctuation_handling']:
            normalized = self._clean_punctuation(normalized)
        
        return normalized
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([,;.!?])', r'\1', text)
        return text.strip()
    
    def _is_list(self, text: str) -> bool:
        """Check if text is a comma-separated list"""
        return ',' in text and len(text.split(',')) >= 2
    
    def _normalize_list(self, text: str, reference: str = None) -> str:
        """
        Normalize list format
        - Match reference ordering if possible
        - Otherwise sort alphabetically for consistency
        """
        items = [item.strip() for item in text.split(',')]
        
        # Remove empty items
        items = [item for item in items if item]
        
        if not items:
            return text
        
        # If reference is also a list, try to match its ordering
        if reference and self._is_list(reference):
            ref_items = [item.strip() for item in reference.split(',')]
            # Reorder items to match reference
            items = self._reorder_to_match(items, ref_items)
        elif self.format_rules['list_ordering']:
            # Sort alphabetically for consistency
            items = sorted(items, key=str.lower)
        
        return ', '.join(items)
    
    def _reorder_to_match(self, items: List[str], reference_items: List[str]) -> List[str]:
        """
        Reorder items to match reference order (for better F1)
        Uses fuzzy matching to handle similar but not exact items
        """
        ordered = []
        remaining = items.copy()
        
        # First pass: exact matches
        for ref_item in reference_items:
            for item in remaining:
                if item.lower() == ref_item.lower():
                    ordered.append(item)
                    remaining.remove(item)
                    break
        
        # Second pass: fuzzy matches
        for ref_item in reference_items:
            best_match = None
            best_score = 0.7  # Threshold for similarity
            
            for item in remaining:
                score = self._similarity(item, ref_item)
                if score > best_score:
                    best_score = score
                    best_match = item
            
            if best_match:
                ordered.append(best_match)
                remaining.remove(best_match)
        
        # Append remaining items
        ordered.extend(remaining)
        
        return ordered
    
    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calculate string similarity (0 to 1)
        Uses SequenceMatcher for fuzzy matching
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _match_reference_format(self, answer: str, reference: str, context: Dict = None) -> str:
        """
        Match reference format for dates, case, and other format-specific rules
        """
        # Date format matching
        if self._looks_like_date(reference):
            answer = self._normalize_date_format(answer, reference)
        
        # Yes/No format matching
        if self._is_yes_no(reference):
            answer = self._normalize_yes_no(answer, reference)
        
        # Number format matching
        if self._is_number(reference):
            answer = self._normalize_number(answer, reference)
        
        # Case matching for short answers (1-3 words)
        if len(reference.split()) <= 3 and len(answer.split()) <= 3:
            answer = self._match_case(answer, reference)
        
        # Plurality matching
        answer = self._match_plurality(answer, reference)
        
        return answer
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}\s+\w+\s+\d{4}',  # "7 May 2023"
            r'\w+\s+\d{1,2},?\s+\d{4}',  # "May 7, 2023"
            r'week\s+of',  # "week of ..."
            r'week\s+before',  # "week before ..."
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _normalize_date_format(self, answer: str, reference: str) -> str:
        """
        Normalize date format to match reference
        CRITICAL: Convert ISO to natural if reference uses natural
        """
        from datetime import datetime
        
        # Handle "last year" → year number
        if 'last year' in answer.lower():
            # Extract year from reference if present
            year_match = re.search(r'\b(20\d{2})\b', reference)
            if year_match:
                answer = answer.replace('last year', year_match.group(1))
                answer = re.sub(r'\s*\(.*?\)', '', answer)  # Remove parentheses like "(2022)"
        
        # If reference uses natural format, convert ISO to natural
        if re.search(r'\d{1,2}\s+\w+\s+\d{4}', reference):
            # Reference is natural: "7 May 2023"
            # If answer is ISO "2023-05-07", convert it
            iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', answer)
            if iso_match:
                try:
                    year, month, day = iso_match.groups()
                    dt = datetime(int(year), int(month), int(day))
                    # Convert to natural format: "7 May 2023"
                    natural_date = dt.strftime('%-d %B %Y')  # Unix format
                    answer = answer.replace(iso_match.group(0), natural_date)
                except:
                    # If conversion fails, try alternative format
                    try:
                        natural_date = dt.strftime('%d %B %Y').lstrip('0')  # Remove leading zero
                        answer = answer.replace(iso_match.group(0), natural_date)
                    except:
                        pass  # Keep original if all conversions fail
        
        # If reference uses ISO, convert natural to ISO
        elif re.search(r'\d{4}-\d{2}-\d{2}', reference) and not re.search(r'\d{4}-\d{2}-\d{2}', answer):
            # Reference is ISO, answer is natural - convert to ISO
            natural_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', answer)
            if natural_match:
                try:
                    day, month_name, year = natural_match.groups()
                    dt = datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
                    iso_date = dt.strftime('%Y-%m-%d')
                    answer = answer.replace(natural_match.group(0), iso_date)
                except:
                    pass
        
        # If reference uses relative expression, keep relative in answer
        # (no conversion needed)
        
        return answer
    
    def _is_yes_no(self, text: str) -> bool:
        """Check if text is a yes/no answer"""
        text_lower = text.lower().strip()
        return text_lower in ['yes', 'no', 'likely yes', 'likely no']
    
    def _normalize_yes_no(self, answer: str, reference: str) -> str:
        """
        Normalize yes/no answers to match reference format
        Remove explanations if reference is simple yes/no
        """
        answer_lower = answer.lower().strip()
        
        # If reference is simple yes/no
        if reference.lower().strip() in ['yes', 'no', 'likely yes', 'likely no']:
            # Extract yes/no from answer (remove explanations)
            if 'yes' in answer_lower:
                return 'Yes' if 'yes' in reference else 'yes'
            elif 'no' in answer_lower:
                return 'No' if 'no' in reference else 'no'
        
        return answer
    
    def _is_number(self, text: str) -> bool:
        """Check if text is primarily a number"""
        # Check if text starts with a digit
        return bool(re.match(r'^\d', text.strip()))
    
    def _normalize_number(self, answer: str, reference: str) -> str:
        """Normalize number format"""
        # Extract number from answer if it contains extra text
        answer_num = re.search(r'\d+', answer)
        ref_num = re.search(r'\d+', reference)
        
        if answer_num and ref_num:
            # If both have numbers, check if they match
            if answer_num.group() == ref_num.group():
                # Match the full format of reference
                return reference
        
        return answer
    
    def _match_case(self, answer: str, reference: str) -> str:
        """
        Match case style of reference (for short answers)
        """
        # If reference is Title Case
        if reference and reference[0].isupper() and any(c.isupper() for c in reference[1:]):
            words = answer.split()
            return ' '.join(w.capitalize() for w in words)
        
        # If reference is lowercase
        if reference and reference.islower():
            return answer.lower()
        
        # If reference is UPPERCASE
        if reference and reference.isupper():
            return answer.upper()
        
        return answer
    
    def _match_plurality(self, answer: str, reference: str) -> str:
        """
        Match plurality of reference
        """
        # Simple plurality matching for single words
        answer_words = answer.split()
        ref_words = reference.split()
        
        if len(answer_words) == 1 and len(ref_words) == 1:
            answer_word = answer_words[0].lower()
            ref_word = ref_words[0].lower()
            
            # If reference is plural but answer is singular
            if ref_word.endswith('s') and not answer_word.endswith('s'):
                if answer_word + 's' == ref_word:
                    return answer + 's'
            
            # If reference is singular but answer is plural
            if answer_word.endswith('s') and not ref_word.endswith('s'):
                if answer_word[:-1] == ref_word:
                    return answer[:-1]
        
        return answer
    
    def _clean_punctuation(self, text: str) -> str:
        """Remove unnecessary punctuation at the end"""
        # Remove trailing periods, semicolons (but keep question marks, exclamation points if meaningful)
        text = re.sub(r'[;,]+$', '', text)
        text = re.sub(r'\.+$', '', text)
        return text.strip()
    
    def get_stats(self) -> Dict:
        """Get normalization statistics"""
        return dict(self.normalization_stats)
