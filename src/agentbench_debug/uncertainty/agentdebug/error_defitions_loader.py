#!/usr/bin/env python3
"""
Unified Error Definitions Loader for AgentBench + AgentDebug

- Preserves original API from ErrorDefinitionsLoader:
    - get_module_definitions(module_name)
    - format_for_phase1_prompt(module_name)
    - format_for_phase2_prompt()
    - get_valid_error_types(module_name)
    - get_all_modules()

- Adds robust normalization utilities so callers may use either snake_case
  (e.g., "inefficient_plan") or PascalCase (e.g., "InefficientPlan").

- Provides "no_error" token for each module, like the original.

Python 3.8+
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

NO_ERROR = "no_error"

# Canonical style we store internally for error-type keys:
# We'll store everything in snake_case internally and expose helpers
# to convert to PascalCase as needed.
def _to_snake(name: str) -> str:
    if not name:
        return name
    s = name.strip()
    # Already snake?
    if "_" in s and s.lower() == s:
        return s
    # Convert common PascalCase → snake_case
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i > 0 and (s[i-1].islower() or (i+1 < len(s) and s[i+1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out).replace("__", "_")

def _to_pascal(snake: str) -> str:
    if not snake:
        return snake
    parts = snake.strip().split("_")
    return "".join(p.capitalize() for p in parts if p)

class ErrorDefinitionsLoader:
    """Loads and manages error type definitions for prompts, with style normalization."""

    def __init__(self):
        self.definitions = self._load_definitions()           # snake_case keys
        self._module_order = ["memory", "reflection", "planning", "action", "system", "others"]

        # Build reverse index: error_type (snake) -> module
        self._error_to_module: Dict[str, str] = {}
        for mod, table in self.definitions.items():
            for etype in table.keys():
                self._error_to_module[etype] = mod
        self._error_to_module[NO_ERROR] = "others"  # benign default

    # -------------------------
    # Core loader (snake_case)
    # -------------------------
    def _load_definitions(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load error definitions with English text (canonical snake_case keys)."""
        definitions: Dict[str, Dict[str, Dict[str, str]]] = {
            'memory': {},
            'reflection': {},
            'planning': {},
            'action': {},
            'system': {},
            'others': {}
        }

        # Memory
        definitions['memory'] = {
            'over_simplification': {
                'definition': (
                    'Agent oversimplifies complex information from previous steps, ignoring details and key factors, '
                    'leading to decisions based on partial or oversimplified summaries.'
                ),
                'example': (
                    'Agent reduces multi-criteria product selection to just "item found", ignoring price/features/inventory.'
                )
            },
            'memory_retrieval_failure': {
                'definition': (
                    'Relevant information exists in agent memory but is not retrieved when needed.'
                ),
                'example': (
                    'Observed a knife on the kitchen countertop at step 3; at step 10, fails to recall it and searches other rooms.'
                )
            },
            'hallucination': {
                'definition': (
                    'Agent "recalls" events or states that never occurred and uses them as a basis for reasoning.'
                ),
                'example': (
                    'Claims "I remember a knife in drawer 1" despite never opening that drawer successfully.'
                )
            }
        }

        # Reflection
        definitions['reflection'] = {
            'progress_misjudge': {
                'definition': (
                    'Agent incorrectly evaluates progress toward the overall goal (overly optimistic or pessimistic).'
                ),
                'example': (
                    'Says "nearly done" before finding a required object; or claims "no progress" after success.'
                )
            },
            'outcome_misinterpretation': {
                'definition': (
                    'Agent executes an action but misinterprets the environment feedback/result.'
                ),
                'example': (
                    'Put(Apple, Microwave) failed due to distance, but agent believes the apple was placed inside.'
                )
            },
            'causal_misattribution': {
                'definition': (
                    'Agent identifies a failure but assigns the wrong cause.'
                ),
                'example': (
                    'Cannot pick up key because it is in a locked safe; agent blames robot arm instead of the locked safe.'
                )
            },
            'hallucination': {
                'definition': (
                    'Agent believes it performed actions that never occurred.'
                ),
                'example': (
                    'Treats a planned step as already executed in a later reflection.'
                )
            }
        }

        # Planning
        definitions['planning'] = {
            'constraint_ignorance': {
                'definition': (
                    'Plan ignores constraints (budget/time/round limits/affordances).'
                ),
                'example': (
                    'Budget is $40 but selects a $55 product; ignores step-limit constraints.'
                )
            },
            'impossible_action': {
                'definition': (
                    'Plan includes actions that are physically or logically impossible in current conditions.'
                ),
                'example': (
                    'Slice(Desk) with a knife; Put(Mug, Sink) while inventory is empty.'
                )
            },
            'inefficient_plan': {
                'definition': (
                    'Plan could succeed but is excessively long, circuitous, or illogical.'
                ),
                'example': (
                    'Goes through multiple irrelevant rooms rather than moving directly to the goal location.'
                )
            },
            'incoherent_subgoals': {
                'definition': (
                    'Plan decomposes into subgoals that do not logically support the main goal or are out of order.'
                ),
                'example': (
                    'Attempts to perform dependent steps without required prerequisites.'
                )
            }
        }

        # Action
        definitions['action'] = {
            'misalignment': {
                'definition': (
                    'Concrete action contradicts current plan/intent.'
                ),
                'example': (
                    'Plan: slice the apple with the knife; Action: GoTo(Bedroom).'
                )
            },
            'invalid_action': {
                'definition': (
                    'Action does not exist or is not allowed.'
                ),
                'example': (
                    'Calls an action outside the available action list.'
                )
            },
            'format_error': {
                'definition': (
                    'Action format invalid; cannot be parsed by the executor.'
                ),
                'example': (
                    'click"product" instead of click["product"].'
                )
            },
            'parameter_error': {
                'definition': (
                    'Action parameters are missing, malformed, or unreasonable.'
                ),
                'example': (
                    'search[query repeated 100 times]; wrong slots; wrong types.'
                )
            }
        }

        # System
        definitions['system'] = {
            'step_limit': {
                'definition': (
                    'Trajectory fails by hitting maximum step budget despite reasonable progress.'
                ),
                'example': (
                    'Two-item task: one placed, searching for second when step limit reached.'
                )
            },
            'tool_execution_error': {
                'definition': (
                    'External tool/API error or unpredictable behavior causes failure.'
                ),
                'example': (
                    'Vision tool mislabels apple as tomato; downstream steps fail.'
                )
            },
            'llm_limit': {
                'definition': (
                    'Failure due to LLM constraints (token limits/timeouts/etc.).'
                ),
                'example': (
                    'Max token exceeded; API timeout; generation abruptly cut.'
                )
            },
            'environment_error': {
                'definition': (
                    'Simulator/environment/network bug or inconsistency contradicts expected rules.'
                ),
                'example': (
                    'Open(Drawer) is valid but engine crashes or object disappears.'
                )
            },
            'context_limit': {
                'definition': (
                    'Failure due to context window overflow/insufficient context capacity.'
                ),
                'example': (
                    'History too long; crucial past info is dropped or truncated.'
                )
            }
        }

        # Others
        definitions['others'] = {
            'others': {
                'definition': 'Problems not covered by other categories.',
                'example': 'Uncategorized anomalies.'
            }
        }

        return definitions

    # -------------------------
    # Public API (original surface)
    # -------------------------
    def get_module_definitions(self, module_name: str) -> Dict[str, Dict[str, str]]:
        """Return snake_case → {definition, example} for a module."""
        return self.definitions.get(module_name, {})

    def format_for_phase1_prompt(self, module_name: str) -> str:
        """
        Format error definitions for Phase 1 prompts.
        (Keeps original output shape; uses snake_case labels + 'no_error')
        """
        module_defs = self.get_module_definitions(module_name)
        if not module_defs:
            if module_name == 'system':
                module_defs = self.definitions.get('system', {})
            elif module_name == 'others':
                return "• **others**: All remaining problems not previously defined or discussed\n"
            else:
                return f"No error definitions found for module: {module_name}"

        formatted = f"DETAILED ERROR TYPE DEFINITIONS FOR {module_name.upper()} MODULE:\n\n"
        for error_type, details in module_defs.items():
            formatted += f"• **{error_type}**:\n"
            formatted += f"  - Definition: {details.get('definition', '')}\n"
            ex = details.get('example')
            if ex:
                formatted += f"  - Example: {ex}\n"
            formatted += "\n"
        formatted += f"• **{NO_ERROR}**: No error detected in this module\n"
        return formatted

    def format_for_phase2_prompt(self) -> str:
        """
        Format all error definitions for Phase 2 critical error identification.
        (Keeps original tone; lists snake_case types with definitions.)
        """
        reference = "COMPLETE ERROR TYPE REFERENCE WITH DEFINITIONS:\n\n"
        for module in self._module_order:
            reference += f"━━━ {module.upper()} MODULE ERRORS ━━━\n"
            module_defs = self.definitions.get(module, {})
            for error_type, details in module_defs.items():
                reference += f"• {error_type}: {details.get('definition', '')}\n"
            reference += "\n"
        return reference

    def get_valid_error_types(self, module_name: str) -> List[str]:
        """Return valid snake_case error types for a module (plus 'no_error')."""
        module_defs = self.get_module_definitions(module_name)
        if module_name == 'others':
            return ['others', NO_ERROR]
        return list(module_defs.keys()) + [NO_ERROR]

    def get_all_modules(self) -> List[str]:
        """Return all module names."""
        return list(self.definitions.keys())

    # -------------------------
    # Extras (quality-of-life)
    # -------------------------
    def normalize_error_type(self, label: str, out_style: str = "snake") -> str:
        """
        Normalize an error label to 'snake' or 'pascal'.
        Accepts snake_case or PascalCase as input.
        """
        s = _to_snake(label or "")
        return s if out_style == "snake" else _to_pascal(s)

    def module_for_error(self, label: str) -> str:
        """
        Return the module name for a given error label (snake or Pascal).
        If unknown, returns '' (empty string).
        """
        s = _to_snake(label or "")
        return self._error_to_module.get(s, "")

    def is_valid_for_module(self, module_name: str, label: str) -> bool:
        """
        True if 'label' is a valid error for 'module_name'.
        Accepts snake or Pascal input.
        """
        s = _to_snake(label or "")
        return s in self.get_valid_error_types(module_name)

    def valid_types(self, module_name: str, style: str = "snake") -> List[str]:
        """
        Convenience: return valid types in requested style ('snake' or 'pascal').
        """
        items = self.get_valid_error_types(module_name)
        if style == "pascal":
            return [_to_pascal(x) for x in items]
        return items
