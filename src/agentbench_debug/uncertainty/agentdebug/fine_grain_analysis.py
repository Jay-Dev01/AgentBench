#!/usr/bin/env python3
"""
AgentBench Unified Fine-Grained Analysis (Phase 1)
- Compatible with AgentBench trajectory dumps (messages/chat_history + metadata)
- Uses ErrorDefinitionsLoader for a single source of truth on error taxonomy
- Produces a normalized Phase-1 artifact that Phase-2 (critical error detection) can consume

Public API:
  - FineGrainedAnalyzer(api_config).process_file(input_path, output_dir, output_filename=None)
  - FineGrainedAnalyzer(api_config).analyze_trajectory(parsed_trajectory_dict)

Author: unified from AgentDebug fine_grained_analysis.py (V5) for AgentBench
"""

import json
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# IMPORTANT: import your unified loader (you added earlier)
from agentbench_debug.error_definitions_loader import ErrorDefinitionsLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# Data models (unchanged in spirit, clarified)
# ------------------------------
@dataclass
class ModuleError:
    module_name: str
    error_type: str
    error_detected: bool
    evidence: str
    reasoning: str


@dataclass
class StepAnalysis:
    step: int
    memory_error: Optional[ModuleError]
    reflection_error: Optional[ModuleError]
    planning_error: Optional[ModuleError]
    action_error: Optional[ModuleError]
    step_summary: str


# ------------------------------
# Fine-grained analyzer
# ------------------------------
class FineGrainedAnalyzer:
    """
    Phase 1: detect module-level errors per step.
    Output schema is the canonical input for Phase 2 (critical error detection).
    """

    def __init__(self, api_config: Dict[str, Any]):
        """
        api_config = {
          "api_key": "...",
          "base_url": "https://api.openai.com/v1/chat/completions" (or equivalent),
          "model": "gpt-4o" (or equivalent),
          "temperature": 0.0,
          "timeout": 60,
          "max_retries": 3
        }
        """
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json",
        }
        self.loader = ErrorDefinitionsLoader()

        # cache of valid types (for reference / future validation if needed)
        self.valid_types = {
            m: self.loader.get_valid_error_types(m)
            for m in self.loader.get_all_modules()
        }

    # ---------- I/O parsing ----------
    def parse_trajectory(self, file_path: str) -> Dict[str, Any]:
        """Parse an AgentBench-style trajectory JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        md = data.get("metadata", {})
        # both legacy and new logs are supported
        messages = data.get("messages", data.get("chat_history", []))

        # best-effort task description extraction (fallback to metadata)
        task_description = md.get("task", "") or self._extract_task_from_messages(messages)

        steps: List[Dict[str, Any]] = []
        step_num = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                step_num += 1
                env_resp = messages[i + 1]["content"] if i + 1 < len(messages) and messages[i + 1].get("role") == "user" else ""
                prev_user = messages[i - 1]["content"] if i - 1 >= 0 and messages[i - 1].get("role") == "user" else ""
                steps.append({
                    "step": step_num,
                    "content": msg.get("content", ""),
                    "env_response": env_resp,
                    "current_input": prev_user
                })

        return {
            "task_id": md.get("task_id", "unknown"),
            "task_description": task_description,
            "success": md.get("success", md.get("won", False)),
            "steps": steps,
            "total_steps": step_num,
            "environment": md.get("environment", "alfworld")
        }

    @staticmethod
    def _extract_task_from_messages(messages: List[Dict[str, Any]]) -> str:
        """Heuristic task extraction from user messages."""
        for m in messages:
            if m.get("role") == "user":
                text = m.get("content", "")
                # Try common “Your task is:” pattern first
                if "Your task is:" in text:
                    return text.split("Your task is:")[1].split("\n")[0].strip()
                # Fall back to first imperative-like sentence
                for line in text.splitlines():
                    if any(k in line.lower() for k in ["task", "put", "find", "book", "buy", "query"]):
                        return line.strip()
        return ""

    # ---------- content extraction ----------
    def extract_modules_from_content(self, content: str, env: str) -> Dict[str, str]:
        """
        Extract <memory>, <reflection>, <plan>, <action> blocks.
        Adds small env-specific adjustments (WebShop, GAIA).
        """
        modules: Dict[str, str] = {}

        def tag(name: str) -> str:
            m = re.search(fr"<{name}>(.*?)</{name}>", content, re.DOTALL | re.IGNORECASE)
            return (m.group(1) if m else "").strip()

        # Base tags
        modules["memory"] = tag("memory")
        modules["reflection"] = tag("reflection")
        modules["planning"] = tag("plan")
        modules["action"] = tag("action")

        # WebShop: pull canonical command if present
        if env.lower() == "webshop" and modules["action"]:
            for pattern in [r"search\[[^\]]*\]", r"click\[[^\]]*\]", r"buy\[[^\]]*\]"]:
                m = re.search(pattern, modules["action"], re.IGNORECASE)
                if m:
                    modules["action"] = m.group(0)
                    break
            # allow <think> as planning if <plan> is missing
            if not modules["planning"]:
                think = tag("think")
                if think:
                    modules["planning"] = think

        # GAIA: tool_call/answer/memory_recall variations
        if env.lower() == "gaia":
            if not modules["memory"]:
                mem_alt = tag("memory_recall")
                if mem_alt:
                    modules["memory"] = mem_alt
            tool_call = tag("tool_call")
            if tool_call:
                modules["action"] = tool_call.strip()
            ans = tag("answer")
            if ans:
                modules["action"] = f"answer: {ans.strip()[:200]}"

        return modules

    def extract_module_content_from_step(self, content: str, module_name: str, env: str) -> str:
        return self.extract_modules_from_content(content, env).get(module_name, "")

    # ---------- system-level checks ----------
    async def _check_system_errors(
        self,
        step_num: int,
        env_response: str
    ) -> Optional[ModuleError]:
        """Cheap guards + optional LLM verification for system errors."""
        # Common step limit heuristic for text envs
        if step_num >= 30:
            return ModuleError(
                module_name="system",
                error_type="step_limit",
                error_detected=True,
                evidence=f"Reached step {step_num}",
                reasoning="Exceeded max allowed steps"
            )

        # Quick scan for obviously system-ish traces
        if env_response and any(k in env_response.lower() for k in ["error", "exception", "timeout", "crash", "stack trace"]):
            prompt = f"""
Is the following an agent-side reasoning error or a SYSTEM error?

Environment Response:
{env_response[:500]}

System error types:
- tool_execution_error (external tool/API failed)
- llm_limit (timeout/max tokens)
- environment_error (simulator/infrastructure crash)

Return JSON:
{{
  "error_detected": true/false,
  "error_type": "tool_execution_error|llm_limit|environment_error|no_error",
  "evidence": "...",
  "reasoning": "..."
}}
"""
            try:
                raw = await self._call_llm(prompt, system_role=(
                    "You are a precise labeler. Respond ONLY with JSON; "
                    "pick a system error if and only if the text indicates a tool/infra failure."
                ))
                parsed = self._parse_detection_json(raw)
                return ModuleError(
                    module_name="system",
                    error_type=parsed.get("error_type", "no_error"),
                    error_detected=bool(parsed.get("error_detected", False)),
                    evidence=parsed.get("evidence", "n/a"),
                    reasoning=parsed.get("reasoning", "n/a"),
                )
            except Exception as e:
                logger.debug("System check LLM fallback due to: %s", e)

        return None

    # ---------- per-module detection ----------
    async def _detect_module_error(
        self,
        module_name: str,
        module_content: str,
        step_num: int,
        step_data: Dict[str, Any],
        task_description: str,
        env_response: str,
        current_step_input: str,
        environment: str
    ) -> ModuleError:
        # Build context stack based on module dependencies (same-step outputs)
        context = f"Current Step Input (user says + history):\n{current_step_input}\n\n"
        if module_name in ("reflection", "planning", "action"):
            mem = self.extract_module_content_from_step(step_data["content"], "memory", environment)
            if mem and module_name in ("reflection", "planning", "action"):
                context += f"Memory (this step):\n{mem}\n\n"
            refl = self.extract_module_content_from_step(step_data["content"], "reflection", environment)
            if refl and module_name in ("planning", "action"):
                context += f"Reflection (this step):\n{refl}\n\n"
            plan = self.extract_module_content_from_step(step_data["content"], "planning", environment)
            if plan and module_name in ("action",):
                context += f"Plan (this step):\n{plan}\n\n"

        # Definitions for this module
        defs = self.loader.format_for_phase1_prompt(module_name)

        prompt = f"""
You detect module-specific errors using the provided definitions.

TASK: {task_description}
ENV: {environment}
STEP: {step_num}

CONTEXT:
{context}

MODULE UNDER TEST: {module_name}
MODULE OUTPUT:
{module_content if module_content else "No content for this module."}

ENV RESPONSE (post-step):
{(env_response or "No response")[:500]}

{defs}

Return ONLY JSON:
{{
  "error_detected": true/false,
  "error_type": "one_of_listed_or_no_error",
  "evidence": "...",
  "reasoning": "..."
}}
"""
        raw = await self._call_llm(
            prompt,
            system_role=(
                "You are an exacting error classifier. "
                "Use ONLY the provided definitions. Respond with JSON ONLY."
            )
        )
        parsed = self._parse_detection_json(raw)

        return ModuleError(
            module_name=module_name,
            error_type=parsed.get("error_type", "unknown"),
            error_detected=bool(parsed.get("error_detected", False)),
            evidence=parsed.get("evidence", "No evidence provided"),
            reasoning=parsed.get("reasoning", "No reasoning provided"),
        )

    # ---------- step & trajectory ----------
    async def analyze_step(
        self,
        step_data: Dict[str, Any],
        task_description: str,
        environment: str
    ) -> StepAnalysis:
        step = step_data["step"]
        content = step_data["content"]
        env_response = step_data["env_response"]
        current_input = step_data.get("current_input", "")

        modules = self.extract_modules_from_content(content, environment)
        errors: Dict[str, Optional[ModuleError]] = {"memory": None, "reflection": None, "planning": None, "action": None}

        # Check system first (applies to the entire step)
        system_err = await self._check_system_errors(step, env_response)
        if system_err and system_err.error_detected:
            # Attach a synthetic “system” line into summary; Phase 2 will still read per-module keys.
            errors["system"] = system_err  # side channel (not in dataclass fields)

        # Skip memory/reflection on step 1 (no prior history to recall/reflect)
        for module_name in ["memory", "reflection", "planning", "action"]:
            if step == 1 and module_name in {"memory", "reflection"}:
                continue
            # If no content (common in some envs), skip detection
            if module_name in {"memory", "reflection"} and not modules.get(module_name):
                continue

            errors[module_name] = await self._detect_module_error(
                module_name=module_name,
                module_content=modules.get(module_name, ""),
                step_num=step,
                step_data=step_data,
                task_description=task_description,
                env_response=env_response,
                current_step_input=current_input,
                environment=environment
            )

        # Step summary text
        found = []
        for name in ["memory", "reflection", "planning", "action"]:
            err = errors.get(name)
            if err and err.error_detected:
                found.append(f"{name}:{err.error_type}")
        # also include system (not part of StepAnalysis dataclass fields)
        sys_err = errors.get("system")
        if sys_err and isinstance(sys_err, ModuleError) and sys_err.error_detected:
            found.append(f"system:{sys_err.error_type}")

        summary = f"Step {step}: " + (", ".join(found) if found else "No errors detected")

        return StepAnalysis(
            step=step,
            memory_error=errors.get("memory"),
            reflection_error=errors.get("reflection"),
            planning_error=errors.get("planning"),
            action_error=errors.get("action"),
            step_summary=summary
        )

    async def analyze_trajectory(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns Phase-1 artifact:
        {
          task_id, task_description, task_success, environment, total_steps,
          step_analyses: [
            {
              step: int,
              errors: { module: {error_type, error_detected, evidence, reasoning}, ... },
              summary: str
            }, ...
          ]
        }
        """
        step_objs: List[StepAnalysis] = []
        for s in traj["steps"]:
            step_objs.append(
                await self.analyze_step(
                    step_data=s,
                    task_description=traj["task_description"],
                    environment=traj["environment"]
                )
            )

        # Serialize for Phase 2
        serialized_steps: List[Dict[str, Any]] = []
        for sa in step_objs:
            item = {
                "step": sa.step,
                "errors": {},
                "summary": sa.step_summary
            }
            for m in ["memory", "reflection", "planning", "action"]:
                err = getattr(sa, f"{m}_error")
                if err:
                    item["errors"][m] = {
                        "error_type": err.error_type,
                        "error_detected": err.error_detected,
                        "evidence": err.evidence,
                        "reasoning": err.reasoning
                    }
            serialized_steps.append(item)

        return {
            "task_id": traj["task_id"],
            "task_description": traj["task_description"],
            "task_success": traj["success"],
            "environment": traj["environment"],
            "total_steps": traj["total_steps"],
            "step_analyses": serialized_steps
        }

    # ---------- driver ----------
    async def process_file(self, input_path: str, output_dir: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """Parse -> analyze -> write Phase-1 JSON artifact."""
        traj = self.parse_trajectory(input_path)
        out = await self.analyze_trajectory(traj)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ofile = Path(output_dir) / (f"{output_filename}_error_detection.json" if output_filename else f"{Path(input_path).stem}_error_detection.json")
        with open(ofile, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        logger.info("Fine-grained analysis: %s | steps=%d | success=%s",
                    out["task_id"], out["total_steps"], out["task_success"])
        return out

    # ---------- LLM helpers ----------
    async def _call_llm(self, prompt: str, system_role: str) -> str:
        payload = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.get("temperature", 0.0),
            "response_format": {"type": "json_object"}
        }
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

        async with aiohttp.ClientSession() as session:
            last_err = None
            for attempt in range(self.config.get("max_retries", 3)):
                try:
                    async with session.post(
                        self.config["base_url"],
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.get("timeout", 60)),
                        proxy=proxy if proxy else None
                    ) as r:
                        r.raise_for_status()
                        data = await r.json()
                        return data["choices"][0]["message"]["content"]
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError(f"LLM call failed after retries: {last_err}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
            return "\n".join(lines)
        return t

    def _parse_detection_json(self, raw: str) -> Dict[str, Any]:
        """
        Robust JSON extraction from LLM content (handles code fences, trailing commas,
        single JSON object selected among candidates).
        """
        txt = self._strip_code_fences(raw)

        # collect JSON candidates by brace counting
        cands: List[str] = []
        start = txt.find("{")
        while start != -1:
            level = 0
            end = None
            for i in range(start, len(txt)):
                ch = txt[i]
                if ch == "{":
                    level += 1
                elif ch == "}":
                    level -= 1
                    if level == 0:
                        end = i + 1
                        cands.append(txt[start:end])
                        break
            if end is None:
                break
            start = txt.find("{", end)

        if not cands:
            raise ValueError("No JSON object found in LLM response.")

        # try strict JSON, then python-literal fallback
        last_exc: Optional[Exception] = None
        for js in cands:
            try:
                js_clean = re.sub(r",(\s*[}\]])", r"\1", js)  # strip trailing commas
                return json.loads(js_clean)
            except Exception as e1:
                try:
                    import ast
                    pyish = re.sub(r"\btrue\b", "True", js, flags=re.IGNORECASE)
                    pyish = re.sub(r"\bfalse\b", "False", pyish, flags=re.IGNORECASE)
                    pyish = re.sub(r"\bnull\b", "None", pyish, flags=re.IGNORECASE)
                    return ast.literal_eval(pyish)
                except Exception as e2:
                    last_exc = e2
                    continue
        raise ValueError(f"Failed to parse detection JSON: {last_exc}")
