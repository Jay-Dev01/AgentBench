# agentbench/agentdebug/critical_error_detection_unified.py
# Unified, deterministic critical-error detection for AgentBench integration.
# No network calls, no async, no external deps. Python 3.8+.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


# ======= Taxonomy (matches AgentDebug modules / error families) =======

MODULES = ("memory", "reflection", "planning", "action", "system", "others")

ACTION_ERRORS = {"FormatError", "InvalidAction", "ParameterError", "Misalignment"}
PLANNING_ERRORS = {"InefficientPlan", "ConstraintIgnorance", "ImpossibleAction", "IncoherentSubgoals"}
REFLECTION_ERRORS = {"ProgressMisjudge", "OutcomeMisinterpretation", "CausalMisattribution", "OverSimplification"}
MEMORY_ERRORS = {"RetrievalFailure", "Hallucination", "Omission"}
SYSTEM_ERRORS = {"StepLimit", "ContextLimit", "ToolExecutionError", "EnvironmentError", "LlmLimit"}

# AgentBench finish-reason → (module, error_type)
FINISH_REASON_MAP: Dict[str, Tuple[str, str]] = {
    "IF": ("action", "FormatError"),
    "IA": ("action", "InvalidAction"),
    "TLE": ("system", "StepLimit"),
    "CLE": ("system", "ContextLimit"),
    "ToolError": ("system", "ToolExecutionError"),
    "EnvError": ("system", "EnvironmentError"),
    # "Complete" => no terminal error
}


@dataclass
class CriticalError:
    critical_step: int
    critical_module: str
    error_type: str
    root_cause: str
    evidence: str
    correction_guidance: str
    cascading_effects: List[Dict[str, Any]]
    confidence: float


# ======= Core API =======

def detect_critical_error_unified(
    *,
    task_id: str,
    environment: str,
    finish_reason: str,             # AgentBench finish reason, e.g., "TLE","IA","IF","CLE","Complete"
    success: bool,                  # True if task succeeded
    step_analyses: List[Dict[str, Any]],  # Stage-1 per-step dicts (see expected keys below)
    messages: Optional[List[Dict[str, str]]] = None,  # Optional: full chat messages (assistant/user alternating)
) -> Optional[Dict[str, Any]]:
    """
    Deterministic Stage-2 critical error detection that returns a dict matching AgentDebug's JSON.

    EXPECTED step_analyses item (minimally):
      {
        "step": int (1-based),
        "errors": {
           "<module>": {
               "error_detected": bool,
               "error_type": str,      # one of taxonomy where possible
               "evidence": str,
               "reasoning": str
           },
           ...
        }
      }

    messages (optional) is AgentBench-style transcript:
      [{"role": "assistant"|"user", "content": "..."}...]

    RETURNS:
      dict(critical_step, critical_module, error_type, root_cause, evidence, correction_guidance,
           cascading_effects, confidence)
      or None if success==True.
    """
    if success:
        return None

    # 1) Gather candidate errors per step
    candidates: List[Tuple[int, str, str, str, str]] = []  # (t, module, error_type, evidence, reasoning)
    for rec in step_analyses:
        t = int(rec.get("step", 0))
        errs = rec.get("errors", {}) or {}
        for module, info in errs.items():
            if not info or not info.get("error_detected"):
                continue
            # Enforce Step-1 rule: no memory/reflection at step 1
            if t == 1 and module in ("memory", "reflection"):
                continue
            etype = str(info.get("error_type", "") or "")
            evidence = str(info.get("evidence", "") or "")
            reasoning = str(info.get("reasoning", "") or "")
            candidates.append((t, module, etype, evidence, reasoning))

    # 2) If finish_reason is a terminal signal (system/action), try to corroborate earliest matching step
    if finish_reason in FINISH_REASON_MAP:
        term_mod, term_err = FINISH_REASON_MAP[finish_reason]
        # earliest corroborating step among candidates
        for t, m, et, ev, rsn in sorted(candidates, key=lambda x: x[0]):
            if m == term_mod and (et == term_err or _error_family_match(m, et, term_err)):
                return _package_critical(
                    t=t, module=m, error_type=et or term_err,
                    evidence=ev or f"Finish reason suggests {term_mod}:{term_err}",
                    reasoning=rsn,
                    step_analyses=step_analyses
                )
        # no corroborating step: still honor finish reason at last step as system/action root cause
        fallback_t = _last_step(step_analyses)
        return _package_critical(
            t=fallback_t, module=term_mod, error_type=term_err,
            evidence=f"Finish reason={finish_reason} with no corroborating tagged step.",
            reasoning="Terminal category selected from environment result.",
            step_analyses=step_analyses
        )

    # 3) Otherwise, pick earliest error following a principled priority:
    #    system > action > planning > reflection > memory (earliest within each)
    priority = ["system", "action", "planning", "reflection", "memory"]
    buckets: Dict[str, List[Tuple[int, str, str, str]]] = {p: [] for p in priority}

    for t, m, et, ev, rsn in candidates:
        if m in buckets:
            buckets[m].append((t, et, ev, rsn))

    for m in priority:
        if buckets[m]:
            t, et, ev, rsn = sorted(buckets[m], key=lambda x: x[0])[0]
            return _package_critical(
                t=t, module=m, error_type=et or _default_error_for(m),
                evidence=ev, reasoning=rsn, step_analyses=step_analyses
            )

    # 4) If we saw no tagged errors at all, default to last-step planning inefficiency
    t = _last_step(step_analyses)
    return _package_critical(
        t=t, module="planning", error_type="InefficientPlan",
        evidence="No explicit tags; trajectory drifted without achieving checkpoints.",
        reasoning="Fallback heuristic: planning inefficiency at last step.",
        step_analyses=step_analyses
    )


# ======= Helpers =======

def _default_error_for(module: str) -> str:
    if module == "action":
        return "InvalidAction"
    if module == "planning":
        return "InefficientPlan"
    if module == "reflection":
        return "ProgressMisjudge"
    if module == "memory":
        return "Omission"
    if module == "system":
        return "StepLimit"
    return "Unknown"

def _error_family_match(module: str, etype: str, expected: str) -> bool:
    """Loose family equality: treat unknown/missing types as family matches when same module."""
    if not etype:
        return True
    families = {
        "action": ACTION_ERRORS,
        "planning": PLANNING_ERRORS,
        "reflection": REFLECTION_ERRORS,
        "memory": MEMORY_ERRORS,
        "system": SYSTEM_ERRORS,
    }
    fam = families.get(module, set())
    return (etype in fam) and (expected in fam)

def _last_step(step_analyses: List[Dict[str, Any]]) -> int:
    if not step_analyses:
        return 1
    return max(int(s.get("step", 1)) for s in step_analyses)

def _collect_cascade(step_analyses: List[Dict[str, Any]], t0: int) -> List[Dict[str, Any]]:
    """Summarize downstream effects as a tiny cascade list."""
    casc: List[Dict[str, Any]] = []
    for rec in step_analyses:
        t = int(rec.get("step", 0))
        if t <= t0:
            continue
        errs = rec.get("errors", {}) or {}
        # If any module recorded an error at step t, add a compact note
        if any(info and info.get("error_detected") for info in errs.values()):
            modules = [m for m, info in errs.items() if info and info.get("error_detected")]
            casc.append({"step": t, "effect": f"Downstream errors in {', '.join(modules)}"})
    return casc

def _package_critical(
    *, t: int, module: str, error_type: str, evidence: str, reasoning: str,
    step_analyses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # Enforce Step-1 rule again defensively
    if t == 1 and module in ("memory", "reflection"):
        module = "planning"
        error_type = "InefficientPlan"

    root_cause = (
        f"Earliest decisive error in {module} at step {t} "
        f"that set the trajectory on a failing path."
    )

    correction = _suggest_correction(module, error_type)

    return {
        "critical_step": t,
        "critical_module": module,
        "error_type": error_type,
        "root_cause": root_cause,
        "evidence": evidence or "Tagged by Stage-1 analysis / finish reason.",
        "correction_guidance": correction,
        "cascading_effects": _collect_cascade(step_analyses, t),
        # Deterministic confidence: earlier + terminal-consistent errors score higher
        "confidence": _confidence_score(module, error_type, t, step_analyses),
    }

def _confidence_score(module: str, error_type: str, t: int, step_analyses: List[Dict[str, Any]]) -> float:
    # Simple, reproducible scoring:
    # - Earlier steps => higher confidence
    # - System / Action slightly higher (terminal causes and executable mistakes are clearer)
    # - Presence of many downstream errors slightly increases confidence
    last = _last_step(step_analyses)
    pos = max(0, last - (t - 1))
    base = 0.4 + 0.6 * (pos / max(1, last))  # 0.4..1.0

    if module == "system":
        base += 0.05
    elif module == "action":
        base += 0.03

    casc = len(_collect_cascade(step_analyses, t))
    base += min(0.05, 0.01 * casc)

    return float(max(0.0, min(1.0, base)))

def _suggest_correction(module: str, error_type: str) -> str:
    if module == "system":
        if error_type == "StepLimit":
            return "Tighten plans; use fewer steps; summarize state; prefer higher-yield actions."
        if error_type == "ContextLimit":
            return "Compress memory; prune history; externalize notes; avoid verbose CoT."
        if error_type == "ToolExecutionError":
            return "Validate tool availability and parameters; add retry with backoff and guardrails."
        if error_type == "EnvironmentError":
            return "Check environment state; add pre-checks; handle non-determinism robustly."
        if error_type == "LlmLimit":
            return "Reduce prompt complexity; simplify instructions; split tasks."
    if module == "action":
        if error_type == "FormatError":
            return "Follow the exact schema/format; add local validators before issuing actions."
        if error_type == "InvalidAction":
            return "Select only valid actions; verify tool name and availability before execution."
        if error_type == "ParameterError":
            return "Validate args; check required fields; confirm types and ranges; dry-run if possible."
        if error_type == "Misalignment":
            return "Ensure action matches plan and instruction; re-verify goal before acting."
    if module == "planning":
        if error_type == "InefficientPlan":
            return "Switch to minimal plan; remove loops; use constraint-aware, goal-directed steps."
        if error_type == "ConstraintIgnorance":
            return "Enumerate constraints; test feasibility; adjust plan before acting."
        if error_type == "ImpossibleAction":
            return "Check preconditions and tool affordances; choose feasible alternatives."
        if error_type == "IncoherentSubgoals":
            return "Derive subgoals from the main goal; ensure logical ordering and dependencies."
    if module == "reflection":
        if error_type == "ProgressMisjudge":
            return "Add explicit success checks; compare observed state vs. goal after each step."
        if error_type == "OutcomeMisinterpretation":
            return "Parse tool outputs more carefully; verify via secondary signals or validators."
        if error_type == "CausalMisattribution":
            return "Trace true causes; avoid blaming irrelevant factors; re-evaluate evidence."
        if error_type == "OverSimplification":
            return "Preserve critical details; avoid premature conclusions; revisit assumptions."
    if module == "memory":
        if error_type == "RetrievalFailure":
            return "Strengthen retrieval keys; store structured state; rehydrate before planning."
        if error_type == "Hallucination":
            return "Cross-validate remembered facts; avoid confabulation; ask environment for ground truth."
        if error_type == "Omission":
            return "Carry forward essential slots (auth, IDs, constraints) across steps explicitly."
    return "Review the step and apply module-appropriate guardrails and validations."
