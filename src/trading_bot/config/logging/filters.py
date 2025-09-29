"""Logging filters for sensitive data masking."""
import re
import logging
from typing import List, Pattern, Dict, Optional, Any


class SensitiveDataFilter(logging.Filter):
    """
    Filter to mask sensitive data in log messages.
    
    Automatically detects and masks:
    - API keys and secrets
    - Account numbers
    - Email addresses
    - IP addresses
    - Credit card numbers
    - Custom patterns
    """
    
    def __init__(
        self,
        mask_enabled: bool = True,
        mask_char: str = "*",
        preserve_length: bool = False,
        custom_patterns: Optional[List[str]] = None
    ):
        """
        Initialize sensitive data filter.
        
        Args:
            mask_enabled: Enable/disable masking
            mask_char: Character to use for masking
            preserve_length: Preserve original length when masking
            custom_patterns: Additional regex patterns to mask
        """
        super().__init__()
        self.mask_enabled = mask_enabled
        self.mask_char = mask_char
        self.preserve_length = preserve_length
        
        # Predefined patterns
        self._patterns: Dict[str, Pattern] = {
            # API keys (various formats)
            "api_key": re.compile(
                r'(api[_-]?key["\s:=]+)([a-zA-Z0-9_-]{20,})',
                re.IGNORECASE
            ),
            "secret": re.compile(
                r'(secret["\s:=]+)([a-zA-Z0-9_-]{20,})',
                re.IGNORECASE
            ),
            "token": re.compile(
                r'(token["\s:=]+)([a-zA-Z0-9._-]{20,})',
                re.IGNORECASE
            ),
            
            # Account/sensitive numbers
            "account": re.compile(
                r'\b\d{10,16}\b'  # 10-16 digit numbers
            ),
            
            # Email addresses
            "email": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            
            # IP addresses
            "ip": re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            
            # Credit cards (basic pattern)
            "credit_card": re.compile(
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ),
        }
        
        # Add custom patterns
        if custom_patterns:
            for i, pattern in enumerate(custom_patterns):
                self._patterns[f"custom_{i}"] = re.compile(pattern)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and mask sensitive data in log record.
        
        Args:
            record: Log record to filter
        
        Returns:
            True (always allow the record, just mask it)
        """
        if not self.mask_enabled:
            return True
        
        # Mask message
        if hasattr(record, 'msg'):
            record.msg = self._mask_sensitive_data(str(record.msg))
        
        # Mask args if present
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._mask_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        return True
    
    def _mask_sensitive_data(self, text: str) -> str:
        """
        Mask sensitive data in text.
        
        Args:
            text: Text to mask
        
        Returns:
            Masked text
        """
        for pattern_name, pattern in self._patterns.items():
            if pattern_name in ["api_key", "secret", "token"]:
                # For key-value patterns, preserve key and mask value
                text = pattern.sub(
                    lambda m: m.group(1) + self._create_mask(m.group(2)),
                    text
                )
            else:
                # For standalone patterns, mask entire match
                text = pattern.sub(
                    lambda m: self._create_mask(m.group(0)),
                    text
                )
        
        return text
    
    def _create_mask(self, original: str) -> str:
        """
        Create masked version of string.
        
        Args:
            original: Original string
        
        Returns:
            Masked string
        """
        if self.preserve_length:
            # Show first 4 and last 4 characters, mask middle
            if len(original) > 8:
                visible_chars = 4
                masked_length = len(original) - (visible_chars * 2)
                return (
                    original[:visible_chars] +
                    self.mask_char * masked_length +
                    original[-visible_chars:]
                )
            else:
                return self.mask_char * len(original)
        else:
            # Fixed length mask
            return self.mask_char * 8


class DebugUnmaskFilter(logging.Filter):
    """
    Filter that disables masking in debug mode.
    
    Use this filter when you need to see full data for debugging.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize debug unmask filter.
        
        Args:
            debug_mode: If True, disable masking
        """
        super().__init__()
        self.debug_mode = debug_mode
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Allow all records (no filtering)."""
        return True