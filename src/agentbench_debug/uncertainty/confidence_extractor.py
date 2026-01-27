"""Extract confidence scores from LLM API responses."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple


class ConfidenceExtractor:
    """
    Extract confidence scores from various LLM API response formats.
    
    Supports:
    - OpenAI logprobs (token-level probabilities)
    - finish_reason signals (stop, tool_calls, length, content_filter)
    - Semantic hedging detection in content AND tool arguments
    - Action complexity analysis
    - Self-reported confidence
    """
    
    # Hedging phrases that indicate uncertainty
    HEDGING_PHRASES = [
        r"\bi think\b",
        r"\bprobably\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bnot sure\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\buncertain\b",
        r"\bi believe\b",
        r"\bit seems\b",
        r"\bappears to\b",
        r"\blikely\b",
        r"\bunlikely\b",
        r"\btry\b",
        r"\battempt\b",
        r"\bguess\b",
        r"\bassume\b",
    ]
    
    # Confident phrases that indicate certainty
    CONFIDENT_PHRASES = [
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bclearly\b",
        r"\bobviously\b",
        r"\bexactly\b",
        r"\bprecisely\b",
        r"\bmust be\b",
        r"\bwill work\b",
        r"\bthis fixes\b",
        r"\bthis solves\b",
    ]
    
    # Action complexity indicators (for SWE-bench tool calls)
    COMPLEX_ACTIONS = {
        "submit_patch": 0.75,  # Submitting = higher stakes, slightly lower confidence
        "bash_command": 0.80,  # Running commands
        "read_file": 0.90,    # Just reading = high confidence
        "search_code": 0.85,  # Searching = fairly confident
    }
    
    # finish_reason to confidence mapping
    FINISH_REASON_CONFIDENCE = {
        "stop": 0.85,
        "tool_calls": 0.80,
        "function_call": 0.80,
        "length": 0.50,
        "content_filter": 0.30,
    }
    
    DEFAULT_CONFIDENCE = 0.70
    
    def __init__(self, default_confidence: float = 0.70, use_semantic_analysis: bool = True):
        self.default_confidence = default_confidence
        self.use_semantic_analysis = use_semantic_analysis
        self._hedging_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HEDGING_PHRASES
        ]
        self._confident_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONFIDENT_PHRASES
        ]
    
    def extract(
        self,
        raw_response: Optional[Dict[str, Any]],
        content: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Extract confidence from API response with enhanced analysis.
        
        Returns:
            Tuple of (confidence_score, source_description)
        """
        if raw_response is None:
            return self.default_confidence, "default"
        
        # Try logprobs first (most accurate)
        confidence, source = self._extract_from_logprobs(raw_response)
        if source != "default":
            return confidence, source
        
        # For tool calls, analyze the tool arguments for semantic content
        if self.use_semantic_analysis:
            tool_conf, tool_source = self._extract_from_tool_calls(raw_response)
            if tool_source != "default":
                return tool_conf, tool_source
        
        # Try finish_reason (baseline for function calls)
        confidence, source = self._extract_from_finish_reason(raw_response)
        if source != "default":
            # If we have content, adjust based on semantic analysis
            if content and self.use_semantic_analysis:
                semantic_adjustment = self._compute_semantic_adjustment(content)
                adjusted_confidence = max(0.3, min(0.95, confidence + semantic_adjustment))
                if abs(semantic_adjustment) > 0.01:
                    return adjusted_confidence, f"{source}+semantic({semantic_adjustment:+.2f})"
            return confidence, source
        
        # Try semantic analysis on content
        if content:
            confidence, source = self._extract_from_semantics(content)
            if source != "default":
                return confidence, source
        
        return self.default_confidence, "default"
    
    def _extract_from_logprobs(
        self, response: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Extract confidence from OpenAI logprobs."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return self.default_confidence, "default"
            
            choice = choices[0]
            logprobs_data = choice.get("logprobs")
            
            if not logprobs_data:
                return self.default_confidence, "default"
            
            # Handle OpenAI chat completions format
            content_logprobs = logprobs_data.get("content", [])
            if not content_logprobs:
                return self.default_confidence, "default"
            
            # Extract token logprobs
            token_logprobs = []
            for token_data in content_logprobs:
                if isinstance(token_data, dict) and "logprob" in token_data:
                    token_logprobs.append(token_data["logprob"])
            
            if not token_logprobs:
                return self.default_confidence, "default"
            
            # Geometric mean of probabilities
            avg_logprob = sum(token_logprobs) / len(token_logprobs)
            confidence = math.exp(avg_logprob)
            
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence, f"logprobs (n={len(token_logprobs)})"
            
        except (KeyError, TypeError, IndexError):
            return self.default_confidence, "default"
    
    def _extract_from_tool_calls(
        self, response: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Extract confidence from tool call content analysis.
        
        Analyzes:
        - Which tool is being called (complexity)
        - Arguments content for hedging/confidence language
        - Command patterns for risky operations
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return self.default_confidence, "default"
            
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls:
                return self.default_confidence, "default"
            
            # Analyze first tool call
            tool_call = tool_calls[0]
            function = tool_call.get("function", {})
            func_name = function.get("name", "")
            func_args_str = function.get("arguments", "{}")
            
            # Base confidence from action type
            base_confidence = self.COMPLEX_ACTIONS.get(func_name, 0.80)
            
            # Parse arguments and analyze content
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                func_args = {}
            
            # Analyze argument content
            arg_text = " ".join(str(v) for v in func_args.values())
            semantic_adjustment = self._compute_semantic_adjustment(arg_text)
            
            # Additional adjustments based on command patterns
            pattern_adjustment = self._analyze_command_patterns(func_name, func_args)
            
            final_confidence = max(0.3, min(0.95, base_confidence + semantic_adjustment + pattern_adjustment))
            
            source_parts = [f"tool:{func_name}"]
            if abs(semantic_adjustment) > 0.01:
                source_parts.append(f"semantic({semantic_adjustment:+.2f})")
            if abs(pattern_adjustment) > 0.01:
                source_parts.append(f"pattern({pattern_adjustment:+.2f})")
            
            return final_confidence, "+".join(source_parts)
            
        except (KeyError, TypeError, IndexError):
            return self.default_confidence, "default"
    
    def _analyze_command_patterns(
        self, func_name: str, func_args: Dict[str, Any]
    ) -> float:
        """
        Analyze command patterns for confidence adjustment.
        
        Returns adjustment value (-0.2 to +0.1).
        """
        adjustment = 0.0
        
        if func_name == "bash_command":
            command = func_args.get("command", "")
            
            # Risky patterns (lower confidence)
            risky_patterns = [
                (r"\brm\s+-rf\b", -0.15),  # Dangerous delete
                (r"\bsudo\b", -0.10),       # Elevated permissions
                (r"\bkill\b", -0.10),       # Process killing
                (r"\bchmod\b", -0.05),      # Permission changes
                (r"\bpip install\b", -0.05), # Installing packages
            ]
            
            # Safe/confident patterns (higher confidence)
            safe_patterns = [
                (r"\bls\b", +0.05),         # Just listing
                (r"\bcat\b", +0.05),        # Just reading
                (r"\bgrep\b", +0.03),       # Searching
                (r"\bpytest.*-v\b", +0.05), # Running tests with verbose
                (r"\bgit\s+status\b", +0.05), # Git status check
                (r"\bgit\s+diff\b", +0.05),   # Git diff
            ]
            
            for pattern, adj in risky_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    adjustment += adj
            
            for pattern, adj in safe_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    adjustment += adj
        
        elif func_name == "submit_patch":
            patch = func_args.get("patch", "")
            
            # Larger patches = slightly lower confidence
            lines = patch.count('\n')
            if lines > 50:
                adjustment -= 0.10
            elif lines > 20:
                adjustment -= 0.05
            elif lines < 5:
                adjustment += 0.05  # Small, focused patch
        
        return max(-0.20, min(0.10, adjustment))
    
    def _compute_semantic_adjustment(self, text: str) -> float:
        """
        Compute confidence adjustment from semantic analysis.
        
        Returns adjustment value (-0.15 to +0.10).
        """
        if not text:
            return 0.0
        
        # Count hedging phrases
        hedging_count = 0
        for pattern in self._hedging_patterns:
            hedging_count += len(pattern.findall(text))
        
        # Count confident phrases
        confident_count = 0
        for pattern in self._confident_patterns:
            confident_count += len(pattern.findall(text))
        
        # Compute adjustment
        hedging_penalty = min(hedging_count * 0.03, 0.15)
        confidence_bonus = min(confident_count * 0.02, 0.10)
        
        return confidence_bonus - hedging_penalty
    
    def _extract_from_finish_reason(
        self, response: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Extract confidence from finish_reason field."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return self.default_confidence, "default"
            
            finish_reason = choices[0].get("finish_reason", "")
            
            if finish_reason in self.FINISH_REASON_CONFIDENCE:
                confidence = self.FINISH_REASON_CONFIDENCE[finish_reason]
                return confidence, f"finish_reason:{finish_reason}"
            
            return self.default_confidence, "default"
            
        except (KeyError, TypeError, IndexError):
            return self.default_confidence, "default"
    
    def _extract_from_semantics(
        self, content: str
    ) -> Tuple[float, str]:
        """Extract confidence from semantic hedging analysis."""
        if not content:
            return self.default_confidence, "default"
        
        adjustment = self._compute_semantic_adjustment(content)
        
        if abs(adjustment) < 0.01:
            return self.default_confidence, "default"
        
        # Base confidence + adjustment
        confidence = max(0.40, min(0.95, 0.75 + adjustment))
        
        hedging_count = sum(len(p.findall(content)) for p in self._hedging_patterns)
        confident_count = sum(len(p.findall(content)) for p in self._confident_patterns)
        
        return confidence, f"semantic (hedge={hedging_count}, conf={confident_count})"
    
    def extract_self_reported(
        self, content: str
    ) -> Tuple[Optional[float], str]:
        """
        Extract self-reported confidence if the model states it.
        
        Looks for patterns like "confidence: 0.8" or "I'm 80% confident"
        """
        if not content:
            return None, "none"
        
        # Pattern: "confidence: X.XX" or "confidence = X.XX"
        match = re.search(
            r"confidence[:\s=]+([0-9]+(?:\.[0-9]+)?)",
            content,
            re.IGNORECASE
        )
        if match:
            value = float(match.group(1))
            # Handle percentage vs decimal
            if value > 1:
                value = value / 100
            return max(0.0, min(1.0, value)), "self_reported"
        
        # Pattern: "X% confident"
        match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*confident",
            content,
            re.IGNORECASE
        )
        if match:
            value = float(match.group(1)) / 100
            return max(0.0, min(1.0, value)), "self_reported_percent"
        
        return None, "none"


__all__ = ["ConfidenceExtractor"]
