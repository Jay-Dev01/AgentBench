"""
LLM Confidence Extraction Module.

Extracts confidence scores from LLM API responses, including:
- OpenAI logprobs
- Gemini candidate scores
- Self-reported confidence from text
- Token probability distributions

This module bridges the gap between raw LLM outputs and the
uncertainty estimation framework.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ConfidenceSignals:
    """Extracted confidence signals from an LLM response."""
    # Primary confidence score (0-1)
    confidence: float
    
    # Source of confidence
    source: str  # "logprobs", "candidate_scores", "self_report", "semantic", "default"
    
    # Token-level details (if available)
    token_logprobs: Optional[List[float]] = None
    mean_logprob: Optional[float] = None
    min_logprob: Optional[float] = None
    perplexity: Optional[float] = None
    
    # Candidate-level details (for APIs that return multiple candidates)
    candidate_scores: Optional[List[float]] = None
    top_candidate_margin: Optional[float] = None
    
    # Self-reported confidence (extracted from text)
    self_reported_confidence: Optional[float] = None
    uncertainty_phrases: List[str] = field(default_factory=list)
    certainty_phrases: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceExtractor:
    """
    Extract confidence scores from various LLM API response formats.
    
    Supports:
    - OpenAI Chat Completions (with logprobs)
    - OpenAI Legacy Completions
    - Google Gemini
    - Anthropic Claude
    - Generic text-based extraction
    """
    
    # Phrases indicating uncertainty
    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i'm uncertain",
        "i don't know",
        "i think",
        "it might be",
        "it could be",
        "possibly",
        "probably",
        "maybe",
        "perhaps",
        "not certain",
        "unclear",
        "unsure",
        "might not",
        "may not",
        "hard to say",
        "difficult to determine",
        "i believe",
        "it seems",
        "appears to",
        "if i had to guess",
    ]
    
    # Phrases indicating certainty
    CERTAINTY_PHRASES = [
        "definitely",
        "certainly",
        "absolutely",
        "i'm sure",
        "i'm certain",
        "without doubt",
        "clearly",
        "obviously",
        "undoubtedly",
        "for certain",
        "positively",
        "i know",
        "it is",
        "the answer is",
        "100%",
        "guaranteed",
        "confirmed",
    ]
    
    def __init__(
        self,
        default_confidence: float = 0.7,
        logprob_scale_factor: float = 1.0,
        enable_semantic_analysis: bool = True,
    ):
        """
        Initialize confidence extractor.
        
        Args:
            default_confidence: Default confidence when no signals available
            logprob_scale_factor: Scale factor for logprob-based confidence
            enable_semantic_analysis: Whether to analyze text for uncertainty phrases
        """
        self.default_confidence = default_confidence
        self.logprob_scale_factor = logprob_scale_factor
        self.enable_semantic_analysis = enable_semantic_analysis
    
    def extract(
        self,
        response: Any,
        api_type: str = "auto",
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """
        Extract confidence from an LLM API response.
        
        Args:
            response: Raw API response (dict, str, or response object)
            api_type: Type of API ("openai", "gemini", "anthropic", "auto")
            content: Optional pre-extracted content for semantic analysis
        
        Returns:
            ConfidenceSignals with extracted confidence information
        """
        # Detect API type if auto
        if api_type == "auto":
            api_type = self._detect_api_type(response)
        
        # Extract based on API type
        if api_type == "openai":
            return self._extract_openai(response, content)
        elif api_type == "openai_legacy":
            return self._extract_openai_legacy(response, content)
        elif api_type == "gemini":
            return self._extract_gemini(response, content)
        elif api_type == "anthropic":
            return self._extract_anthropic(response, content)
        else:
            # Generic extraction
            return self._extract_generic(response, content)
    
    def _detect_api_type(self, response: Any) -> str:
        """Auto-detect API type from response structure."""
        if not isinstance(response, dict):
            return "generic"
        
        # OpenAI Chat Completion
        if "choices" in response and isinstance(response.get("choices"), list):
            if response["choices"] and "message" in response["choices"][0]:
                return "openai"
            elif response["choices"] and "text" in response["choices"][0]:
                return "openai_legacy"
        
        # Gemini
        if "candidates" in response:
            return "gemini"
        
        # Anthropic
        if "content" in response and "type" in response:
            return "anthropic"
        if "completion" in response:
            return "anthropic"
        
        return "generic"
    
    # =========================================================================
    # OpenAI Extraction
    # =========================================================================
    
    def _extract_openai(
        self,
        response: Dict[str, Any],
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """Extract confidence from OpenAI Chat Completion response."""
        signals = ConfidenceSignals(
            confidence=self.default_confidence,
            source="default",
        )
        
        choices = response.get("choices", [])
        if not choices:
            return signals
        
        choice = choices[0]
        message = choice.get("message", {})
        
        # Get content for semantic analysis
        if content is None:
            content = message.get("content", "")
        
        # Check for logprobs
        logprobs_data = choice.get("logprobs")
        if logprobs_data:
            signals = self._process_openai_logprobs(logprobs_data, content)
        else:
            # Fall back to semantic analysis
            signals = self._extract_semantic_confidence(content)
        
        # Check finish reason
        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            # Truncated response - reduce confidence
            signals.confidence *= 0.8
            signals.metadata["truncated"] = True
        
        return signals
    
    def _process_openai_logprobs(
        self,
        logprobs_data: Dict[str, Any],
        content: str,
    ) -> ConfidenceSignals:
        """Process OpenAI logprobs format."""
        # OpenAI Chat Completion logprobs format
        content_logprobs = logprobs_data.get("content", [])
        
        if not content_logprobs:
            return self._extract_semantic_confidence(content)
        
        # Extract token logprobs
        token_logprobs = []
        for token_data in content_logprobs:
            logprob = token_data.get("logprob")
            if logprob is not None:
                token_logprobs.append(logprob)
        
        if not token_logprobs:
            return self._extract_semantic_confidence(content)
        
        # Compute statistics
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
        min_logprob = min(token_logprobs)
        
        # Convert to probability-based confidence
        # Higher logprobs = higher confidence
        # Typical range: -0.1 (very confident) to -5 (uncertain)
        # Map to 0-1 confidence
        
        # Using mean logprob
        # e^-0.1 ≈ 0.90, e^-1 ≈ 0.37, e^-3 ≈ 0.05
        mean_prob = math.exp(mean_logprob)
        
        # Adjust for typical ranges
        confidence = min(1.0, mean_prob * self.logprob_scale_factor)
        
        # Perplexity
        perplexity = math.exp(-mean_logprob)
        
        # Also factor in semantic analysis
        semantic_signals = self._extract_semantic_confidence(content)
        
        # Combine signals (weighted average)
        combined_confidence = (
            0.6 * confidence +
            0.4 * semantic_signals.confidence
        )
        
        return ConfidenceSignals(
            confidence=combined_confidence,
            source="logprobs",
            token_logprobs=token_logprobs,
            mean_logprob=mean_logprob,
            min_logprob=min_logprob,
            perplexity=perplexity,
            self_reported_confidence=semantic_signals.self_reported_confidence,
            uncertainty_phrases=semantic_signals.uncertainty_phrases,
            certainty_phrases=semantic_signals.certainty_phrases,
            metadata={"raw_mean_prob": mean_prob},
        )
    
    def _extract_openai_legacy(
        self,
        response: Dict[str, Any],
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """Extract confidence from OpenAI Legacy Completion response."""
        choices = response.get("choices", [])
        if not choices:
            return ConfidenceSignals(
                confidence=self.default_confidence,
                source="default",
            )
        
        choice = choices[0]
        
        if content is None:
            content = choice.get("text", "")
        
        # Check for logprobs
        logprobs = choice.get("logprobs")
        if logprobs and logprobs.get("token_logprobs"):
            token_logprobs = [lp for lp in logprobs["token_logprobs"] if lp is not None]
            
            if token_logprobs:
                mean_logprob = sum(token_logprobs) / len(token_logprobs)
                min_logprob = min(token_logprobs)
                mean_prob = math.exp(mean_logprob)
                
                return ConfidenceSignals(
                    confidence=min(1.0, mean_prob * self.logprob_scale_factor),
                    source="logprobs",
                    token_logprobs=token_logprobs,
                    mean_logprob=mean_logprob,
                    min_logprob=min_logprob,
                    perplexity=math.exp(-mean_logprob),
                )
        
        return self._extract_semantic_confidence(content)
    
    # =========================================================================
    # Gemini Extraction
    # =========================================================================
    
    def _extract_gemini(
        self,
        response: Dict[str, Any],
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """Extract confidence from Google Gemini response."""
        candidates = response.get("candidates", [])
        
        if not candidates:
            return ConfidenceSignals(
                confidence=self.default_confidence,
                source="default",
            )
        
        # Get primary candidate
        candidate = candidates[0]
        
        # Extract content
        if content is None:
            content_parts = candidate.get("content", {}).get("parts", [])
            content = " ".join(part.get("text", "") for part in content_parts)
        
        # Check for safety ratings
        safety_ratings = candidate.get("safetyRatings", [])
        safety_factor = 1.0
        for rating in safety_ratings:
            if rating.get("probability", "") in ["MEDIUM", "HIGH"]:
                safety_factor *= 0.9  # Reduce confidence for flagged content
        
        # Check finish reason
        finish_reason = candidate.get("finishReason", "")
        truncation_factor = 1.0
        if finish_reason == "MAX_TOKENS":
            truncation_factor = 0.85
        elif finish_reason == "SAFETY":
            truncation_factor = 0.7
        
        # Get candidate scores if multiple candidates
        candidate_scores = []
        if len(candidates) > 1:
            for cand in candidates:
                # Gemini may provide different scoring mechanisms
                score = cand.get("score", cand.get("avgLogprobs", 0))
                candidate_scores.append(score)
        
        # Base confidence from semantic analysis
        semantic_signals = self._extract_semantic_confidence(content)
        
        confidence = semantic_signals.confidence * safety_factor * truncation_factor
        
        return ConfidenceSignals(
            confidence=confidence,
            source="candidate_scores" if candidate_scores else "semantic",
            candidate_scores=candidate_scores if candidate_scores else None,
            top_candidate_margin=(
                candidate_scores[0] - candidate_scores[1]
                if len(candidate_scores) >= 2 else None
            ),
            self_reported_confidence=semantic_signals.self_reported_confidence,
            uncertainty_phrases=semantic_signals.uncertainty_phrases,
            certainty_phrases=semantic_signals.certainty_phrases,
            metadata={
                "finish_reason": finish_reason,
                "safety_factor": safety_factor,
            },
        )
    
    # =========================================================================
    # Anthropic Extraction
    # =========================================================================
    
    def _extract_anthropic(
        self,
        response: Dict[str, Any],
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """Extract confidence from Anthropic Claude response."""
        # Get content
        if content is None:
            if "content" in response:
                # Messages API format
                content_blocks = response.get("content", [])
                content = " ".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                )
            else:
                # Legacy format
                content = response.get("completion", "")
        
        # Check stop reason
        stop_reason = response.get("stop_reason", "")
        truncation_factor = 1.0
        if stop_reason == "max_tokens":
            truncation_factor = 0.85
        
        # Anthropic doesn't provide logprobs, so rely on semantic analysis
        semantic_signals = self._extract_semantic_confidence(content)
        
        confidence = semantic_signals.confidence * truncation_factor
        
        return ConfidenceSignals(
            confidence=confidence,
            source="semantic",
            self_reported_confidence=semantic_signals.self_reported_confidence,
            uncertainty_phrases=semantic_signals.uncertainty_phrases,
            certainty_phrases=semantic_signals.certainty_phrases,
            metadata={"stop_reason": stop_reason},
        )
    
    # =========================================================================
    # Generic/Semantic Extraction
    # =========================================================================
    
    def _extract_generic(
        self,
        response: Any,
        content: Optional[str] = None,
    ) -> ConfidenceSignals:
        """Generic extraction for unknown response formats."""
        if content is None:
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                # Try common keys
                content = (
                    response.get("content") or
                    response.get("text") or
                    response.get("output") or
                    response.get("response") or
                    str(response)
                )
            else:
                content = str(response)
        
        return self._extract_semantic_confidence(content)
    
    def _extract_semantic_confidence(self, content: str) -> ConfidenceSignals:
        """Extract confidence from semantic analysis of text."""
        if not content or not self.enable_semantic_analysis:
            return ConfidenceSignals(
                confidence=self.default_confidence,
                source="default",
            )
        
        content_lower = content.lower()
        
        # Find uncertainty phrases
        uncertainty_found = []
        for phrase in self.UNCERTAINTY_PHRASES:
            if phrase in content_lower:
                uncertainty_found.append(phrase)
        
        # Find certainty phrases
        certainty_found = []
        for phrase in self.CERTAINTY_PHRASES:
            if phrase in content_lower:
                certainty_found.append(phrase)
        
        # Look for self-reported confidence percentages
        self_reported = None
        confidence_patterns = [
            r"(\d{1,3})%\s*(?:confident|sure|certain)",
            r"confidence[:\s]+(\d{1,3})%",
            r"certainty[:\s]+(\d{1,3})%",
            r"i(?:'m|\s+am)\s+(\d{1,3})%\s+(?:confident|sure|certain)",
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content_lower)
            if match:
                try:
                    self_reported = float(match.group(1)) / 100.0
                    break
                except ValueError:
                    pass
        
        # Compute confidence score
        if self_reported is not None:
            # Use self-reported if available
            confidence = self_reported
        else:
            # Compute from phrase analysis
            base_confidence = self.default_confidence
            
            # Adjust for uncertainty phrases
            uncertainty_penalty = len(uncertainty_found) * 0.08
            
            # Adjust for certainty phrases
            certainty_bonus = len(certainty_found) * 0.05
            
            confidence = base_confidence - uncertainty_penalty + certainty_bonus
            
            # Clamp to valid range
            confidence = max(0.1, min(0.95, confidence))
        
        return ConfidenceSignals(
            confidence=confidence,
            source="semantic" if (uncertainty_found or certainty_found or self_reported) else "default",
            self_reported_confidence=self_reported,
            uncertainty_phrases=uncertainty_found,
            certainty_phrases=certainty_found,
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def extract_from_multiple_samples(
        self,
        responses: List[Any],
        api_type: str = "auto",
    ) -> ConfidenceSignals:
        """
        Extract confidence from multiple sampled responses (self-consistency).
        
        Args:
            responses: List of API responses for the same prompt
            api_type: API type for parsing
        
        Returns:
            Combined confidence signals
        """
        if not responses:
            return ConfidenceSignals(
                confidence=self.default_confidence,
                source="default",
            )
        
        # Extract from each response
        all_signals = [self.extract(resp, api_type) for resp in responses]
        
        # Compute agreement-based confidence
        confidences = [s.confidence for s in all_signals]
        mean_confidence = sum(confidences) / len(confidences)
        
        # Check if outputs are consistent (would need content comparison)
        # For now, use confidence variance as proxy
        if len(confidences) > 1:
            variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
            # High variance = disagreement = lower effective confidence
            agreement_factor = max(0.5, 1.0 - variance * 2)
        else:
            agreement_factor = 1.0
        
        combined_confidence = mean_confidence * agreement_factor
        
        # Aggregate token logprobs if available
        all_logprobs = []
        for s in all_signals:
            if s.token_logprobs:
                all_logprobs.extend(s.token_logprobs)
        
        return ConfidenceSignals(
            confidence=combined_confidence,
            source="self_consistency",
            token_logprobs=all_logprobs if all_logprobs else None,
            mean_logprob=sum(all_logprobs) / len(all_logprobs) if all_logprobs else None,
            metadata={
                "n_samples": len(responses),
                "individual_confidences": confidences,
                "agreement_factor": agreement_factor,
            },
        )


def extract_confidence(
    response: Any,
    api_type: str = "auto",
    content: Optional[str] = None,
) -> float:
    """
    Convenience function to extract confidence score.
    
    Args:
        response: LLM API response
        api_type: API type ("openai", "gemini", "anthropic", "auto")
        content: Optional pre-extracted content
    
    Returns:
        Confidence score (0-1)
    """
    extractor = ConfidenceExtractor()
    signals = extractor.extract(response, api_type, content)
    return signals.confidence


__all__ = [
    "ConfidenceExtractor",
    "ConfidenceSignals",
    "extract_confidence",
]

