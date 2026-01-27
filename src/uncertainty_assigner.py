"""
Uncertainty-aware assigner for AgentBench.

This module extends the standard assigner with real-time uncertainty tracking.
It wraps agents with UncertaintyAwareAgent to capture confidence during inference.

Enhanced features:
- Semantic analysis of tool call arguments
- Action complexity scoring
- Task outcome calibration with Spearman correlation
- AUROC discrimination metric

Usage:
    python -m src.uncertainty_assigner --config configs/assignments/your_config.yaml
"""

import datetime
import json
import os
import random
import threading
import time
from typing import Dict, List, Union, Any, Optional
from typing import Tuple, Callable, Iterator
import contextlib
import sys
from tqdm.contrib import DummyTqdmFile

import yaml
from tqdm import tqdm

from src.client.task import TaskError
from .client import TaskClient, AgentClient
from .configs import ConfigLoader
from .typings import AssignmentConfig, SampleIndex, TaskOutput, TaskClientOutput
from .utils import ColorMessage
from .utils import Graph, MaxFlow

# Import uncertainty components
from src.agentbench_debug.uncertainty import (
    UncertaintyAwareAgent,
    UncertaintyTracker,
    UncertaintyReport,
    ConfidenceExtractor,
    StepInfo,
)
from src.agentbench_debug.uncertainty.calibration import (
    CalibrationMetrics,
    CalibrationResult,
    OutcomeAnalysis,
)


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    except Exception as exc:
        raise exc
    finally:
        sys.stdout, sys.stderr = orig_out_err


class UncertaintyAwareAgentWrapper:
    """
    Wrapper that intercepts agent inference to track uncertainty and latency.
    
    Enhanced with:
    - Semantic analysis of tool arguments
    - Action complexity scoring  
    - Hedging/confidence phrase detection
    - Per-step and trajectory-level latency tracking
    """
    
    def __init__(
        self,
        agent: AgentClient,
        task_type: str = "unknown",
        default_confidence: float = 0.70,
        use_semantic_analysis: bool = True,
    ):
        self._agent = agent
        self._task_type = task_type
        # Use enhanced confidence extractor with semantic analysis
        self._confidence_extractor = ConfidenceExtractor(
            default_confidence=default_confidence,
            use_semantic_analysis=use_semantic_analysis,
        )
        self._tracker = UncertaintyTracker(task_type=task_type)
        self._step_count = 0
        self._confidence_history: List[Tuple[float, str]] = []  # (confidence, source)
        self._latency_history: List[float] = []  # Per-step latencies in seconds
        self._trajectory_start_time: Optional[float] = None  # Wall-clock start
        self._default_confidence = default_confidence
    
    @property
    def name(self) -> str:
        return getattr(self._agent, "name", "unknown")
    
    def inference(self, history: Any) -> str:
        """Perform inference with enhanced uncertainty and latency tracking."""
        # Start trajectory timer on first step
        if self._trajectory_start_time is None:
            self._trajectory_start_time = time.time()
        
        # Measure per-step latency
        step_start = time.time()
        
        # Call underlying agent
        output = self._agent.inference(history)
        
        # Record step latency
        step_latency = time.time() - step_start
        self._latency_history.append(step_latency)
        
        # Try to get raw response
        raw_response = None
        if hasattr(self._agent, "get_last_raw_response"):
            raw_response = self._agent.get_last_raw_response()
        elif hasattr(self._agent, "_last_raw_response"):
            raw_response = self._agent._last_raw_response
        
        # Extract confidence using enhanced extractor
        confidence, source = self._confidence_extractor.extract(raw_response, output)
        
        # Store confidence for this step
        self._confidence_history.append((confidence, source))
        self._step_count += 1
        
        # Print step info with source details and latency
        print(f"[Uncertainty] Step {self._step_count}: confidence={confidence:.3f} ({source}), latency={step_latency:.2f}s")
        
        return output
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get detailed summary of tracked uncertainty."""
        if not self._confidence_history:
            return {
                "n_steps": 0,
                "mean_confidence": self._default_confidence,
                "min_confidence": self._default_confidence,
                "trend": "stable",
                "confidence_sources": {},
            }
        
        confs = [c for c, _ in self._confidence_history]
        sources = [s for _, s in self._confidence_history]
        mean_conf = sum(confs) / len(confs)
        
        # Compute trend
        if len(confs) >= 2:
            mid = len(confs) // 2
            first_half = sum(confs[:mid]) / mid if mid > 0 else mean_conf
            second_half = sum(confs[mid:]) / (len(confs) - mid)
            if second_half > first_half + 0.05:
                trend = "increasing"
            elif second_half < first_half - 0.05:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Count confidence sources
        source_counts = {}
        for s in sources:
            # Extract base source type
            base_source = s.split(":")[0].split("+")[0]
            source_counts[base_source] = source_counts.get(base_source, 0) + 1
        
        # Compute variance for uncertainty measure
        variance = sum((c - mean_conf) ** 2 for c in confs) / len(confs) if confs else 0
        
        # Compute latency statistics
        latency_stats = self._compute_latency_stats()
        
        return {
            "n_steps": len(confs),
            "mean_confidence": mean_conf,
            "min_confidence": min(confs),
            "max_confidence": max(confs),
            "confidence_variance": variance,
            "trajectory_uncertainty": 1 - mean_conf,  # As per paper formula
            "trend": trend,
            "confidence_history": confs,
            "confidence_sources": source_counts,
            "high_uncertainty_steps": sum(1 for c in confs if c < 0.5),
            # Latency metrics
            "latency": latency_stats,
        }
    
    def _compute_latency_stats(self) -> Dict[str, Any]:
        """Compute latency statistics for the trajectory."""
        if not self._latency_history:
            return {
                "total_time": 0.0,
                "mean_step_latency": 0.0,
                "min_step_latency": 0.0,
                "max_step_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "step_latencies": [],
            }
        
        latencies = sorted(self._latency_history)
        n = len(latencies)
        
        # Wall-clock time (trajectory end - start)
        if self._trajectory_start_time:
            total_time = time.time() - self._trajectory_start_time
        else:
            total_time = sum(latencies)
        
        # Percentile calculation
        def percentile(p: float) -> float:
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return latencies[f] + (k - f) * (latencies[c] - latencies[f]) if c != f else latencies[f]
        
        return {
            "total_time": total_time,
            "mean_step_latency": sum(latencies) / n,
            "min_step_latency": latencies[0],
            "max_step_latency": latencies[-1],
            "p50_latency": percentile(50),
            "p95_latency": percentile(95),
            "p99_latency": percentile(99),
            "step_latencies": self._latency_history,
        }
    
    def reset(self) -> None:
        """Reset tracker for new run."""
        self._step_count = 0
        self._confidence_history = []
        self._latency_history = []
        self._trajectory_start_time = None
    
    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped agent."""
        return getattr(self._agent, name)


def _detect_task_type(task_name: str) -> str:
    """Detect task type from task name."""
    name_lower = task_name.lower()
    
    if "alfworld" in name_lower:
        return "alfworld"
    elif "dbbench" in name_lower or "db" in name_lower:
        return "dbbench"
    elif "os" in name_lower and "webshop" not in name_lower:
        return "os_interaction"
    elif "kg" in name_lower or "knowledge" in name_lower:
        return "knowledgegraph"
    elif "webshop" in name_lower:
        return "webshop"
    elif "swebench" in name_lower or "swe" in name_lower:
        return "swebench"
    elif "toolemu" in name_lower:
        return "toolemu"
    
    return "unknown"


class UncertaintyAssigner:
    """
    Assigner with uncertainty tracking for all agents.
    
    Enhanced with:
    - Real-time confidence extraction from tool calls
    - Semantic analysis of agent reasoning
    - Task outcome calibration (Spearman correlation)
    - AUROC discrimination metric
    """
    
    def __init__(self, config: AssignmentConfig, auto_retry: bool = True) -> None:
        self.auto_retry = auto_retry
        self.tqdm_ordered_by_agent = {}
        self.overall_tqdm = None
        self.config = config
        self.free_worker = config.concurrency.copy(deep=True)
        self.agents: Dict[str, AgentClient] = {}
        self.wrapped_agents: Dict[str, Dict[str, UncertaintyAwareAgentWrapper]] = {}  # {agent: {task: wrapper}}
        self.tasks: Dict[str, TaskClient] = {}
        self.task_indices: Dict[str, List[SampleIndex]] = {}
        self.task_worker_fail_count: Dict[str, int] = {}
        self.assignment_lock = threading.Lock()
        self.remaining_tasks: Dict[str, Dict[str, List[int]]] = {}
        self.completions: Dict[str, Dict[str, List[TaskOutput]]] = {}
        self.finished_count = 0
        self.started_count = 0
        self.running_count = 0
        
        # Uncertainty tracking
        self.run_summaries: List[Dict[str, Any]] = []
        
        # Calibration metrics
        self.calibration_metrics = CalibrationMetrics(n_bins=10)
        
        # Initialize (same as original Assigner)
        if not os.path.exists(self.config.output):
            os.makedirs(self.config.output)
            with open(os.path.join(self.config.output, "config.yaml"), "w") as f:
                f.write(yaml.dump(self.config.dict()))
        
        # Load existing runs
        for assignment in self.config.assignments:
            agent = assignment.agent
            task = assignment.task
            runs_file = os.path.join(self.get_output_dir(agent, task), "runs.jsonl")
            result_file = os.path.join(self.get_output_dir(agent, task), "overall.json")
            if os.path.exists(result_file):
                continue
            if agent not in self.remaining_tasks:
                self.remaining_tasks[agent] = {}
            if task not in self.remaining_tasks[agent]:
                self.remaining_tasks[agent][task] = []
            if task not in self.tasks:
                print(ColorMessage.green(f"creating {task} client..."))
                self.tasks[task] = self.config.definition.task[task].create()
                self.task_indices[task] = self.tasks[task].get_indices()
            self.remaining_tasks[agent][task] = self.task_indices[task].copy()
            if not os.path.exists(runs_file):
                continue
            with open(runs_file, "r") as f:
                for line in f:
                    try:
                        run = json.loads(line)
                        run.pop("time")
                        index = run.pop("index")
                        assert index is not None
                        run = TaskClientOutput.parse_obj(run)
                        assert isinstance(run.output, TaskOutput)
                    except:
                        continue
                    if index in self.remaining_tasks[agent][task]:
                        self.remaining_tasks[agent][task].remove(index)
                        self.record_completion(agent, task, index, run.output)
                    else:
                        print(ColorMessage.yellow(
                            f"Warning: {agent}/{task}#{index} is finished, but not in the index list."
                        ))
        
        count = sum(
            len(self.remaining_tasks[agent][task])
            for agent in self.remaining_tasks
            for task in self.remaining_tasks[agent]
        )
        print(ColorMessage.cyan(f"Message: {count} samples remaining."))
        
        for agent in self.remaining_tasks:
            tasks_ = len(self.remaining_tasks[agent])
            samples_ = sum(
                len(self.remaining_tasks[agent][task])
                for task in self.remaining_tasks[agent]
            )
            if samples_ == 0:
                continue
            print(ColorMessage.cyan(
                f"Agent {agent} needs to run {tasks_} tasks with {samples_} samples"
            ))
        
        # Create agents with uncertainty wrappers
        for agent in self.remaining_tasks:
            print(ColorMessage.green(f"Creating uncertainty-aware agent: {agent}"))
            base_agent = self.config.definition.agent[agent].create()
            self.agents[agent] = base_agent
            self.wrapped_agents[agent] = {}
            
            # Create wrapped version for each task
            for task in self.remaining_tasks[agent]:
                task_type = _detect_task_type(task)
                self.wrapped_agents[agent][task] = UncertaintyAwareAgentWrapper(
                    agent=base_agent,
                    task_type=task_type,
                    use_semantic_analysis=True,
                )
    
    def get_output_dir(self, agent: str, task: str) -> str:
        return os.path.join(self.config.output, agent, task)
    
    def worker_generator(self, interval=10) -> Iterator[Tuple[str, str, SampleIndex]]:
        node_list = ["SRC", "DST"]
        agent_node_index = {}
        task_node_index = {}
        for agent in self.agents:
            node_list.append(agent)
            agent_node_index[agent] = len(node_list) - 1
        for task in self.tasks:
            node_list.append(task)
            task_node_index[task] = len(node_list) - 1
        
        while True:
            with self.assignment_lock:
                for task in self.tasks:
                    self.free_worker.task[task] = self.tasks[task].get_concurrency()
                print("Running Count: {}".format(self.running_count))
            
            with self.assignment_lock:
                edges = {}
                for agent in self.agents:
                    edges[(0, agent_node_index[agent])] = self.free_worker.agent[agent]
                for task in self.tasks:
                    edges[(task_node_index[task], 1)] = self.free_worker.task[task]
                tot_remaining_samples = 0
                for agent in self.remaining_tasks:
                    for task in self.remaining_tasks[agent]:
                        tot_remaining_samples += len(self.remaining_tasks[agent][task])
                        edges[(agent_node_index[agent], task_node_index[task])] = len(
                            self.remaining_tasks[agent][task]
                        )
            if tot_remaining_samples == 0:
                if self.running_count == 0:
                    break
                else:
                    time.sleep(interval / 2 + random.random() * interval)
                    continue
            
            graph = Graph(node_count=len(node_list), edges=edges)
            max_flow = MaxFlow(graph, src=0, dst=1)
            
            if max_flow.max_flow == 0:
                time.sleep(interval / 2 + random.random() * interval)
                continue
            
            for (src, dst), e in max_flow.edges_dict.items():
                if src not in agent_node_index.values() or dst not in task_node_index.values():
                    continue
                if e.flow == 0:
                    continue
                agent = node_list[src]
                task = node_list[dst]
                for _ in range(e.flow):
                    with self.assignment_lock:
                        index = self.remaining_tasks[agent][task].pop()
                        self.free_worker.agent[agent] -= 1
                        self.free_worker.task[task] -= 1
                    print(ColorMessage.green(f"Assigned {agent}/{task}#{index}"))
                    yield agent, task, index
            
            time.sleep(interval / 2 + random.random() * interval)
    
    def start(self, tqdm_out=None):
        print("\n" + "=" * 60)
        print("UNCERTAINTY-AWARE AGENTBENCH EVALUATION")
        print("=" * 60)
        print("This run will track agent confidence in real-time.")
        print("Enhanced with semantic analysis and calibration metrics.")
        print("=" * 60 + "\n")
        
        self.started_count = sum(
            len(self.remaining_tasks[agent][task])
            for agent in self.remaining_tasks
            for task in self.remaining_tasks[agent]
        )
        generator = self.worker_generator()
        self.overall_tqdm = tqdm(
            total=self.started_count,
            desc="Total",
            position=0,
            file=tqdm_out,
        )
        for idx, agent in enumerate(self.remaining_tasks.keys()):
            self.tqdm_ordered_by_agent[agent] = tqdm(
                total=sum(
                    len(self.remaining_tasks[agent][task])
                    for task in self.remaining_tasks[agent]
                ),
                desc=agent,
                position=idx + 1,
                file=tqdm_out,
            )
        
        while True:
            try:
                agent, task, index = next(generator)
            except StopIteration:
                break
            self.start_worker(agent, task, index, self.finish_callback)
        
        self.overall_tqdm.close()
        for agent in self.tqdm_ordered_by_agent:
            self.tqdm_ordered_by_agent[agent].close()
        
        # Save uncertainty analysis with calibration
        self._save_uncertainty_analysis()
        
        final_message = (
            "\n\n============================================\n"
            + ColorMessage.cyan(f"Message: {self.started_count} sample(s) started. ")
            + "\n"
            + ColorMessage.green(f"   >> {self.finished_count} sample(s) finished successfully.")
            + "\n"
        )
        if self.started_count != self.finished_count:
            final_message += (
                ColorMessage.red(f"   >> {self.started_count - self.finished_count} sample(s) failed.")
                + "\n"
            )
        final_message += (
            ColorMessage.cyan(f"   >> results are saved to {self.config.output}")
            + "\n"
            + ColorMessage.cyan(f"   >> uncertainty analysis saved to {self.config.output}/uncertainty_analysis.json")
            + "\n"
        )
        final_message += "============================================\n\n"
        print(final_message)
    
    def _save_uncertainty_analysis(self):
        """Save uncertainty analysis to file with calibration and latency metrics."""
        if not self.run_summaries:
            return
        
        # Extract confidences and outcomes for calibration
        confidences = [r.get("mean_confidence", 0.7) for r in self.run_summaries]
        outcomes = [r.get("success", False) for r in self.run_summaries]
        
        # Compute calibration metrics
        calibration_result = self.calibration_metrics.compute(confidences, outcomes)
        outcome_analysis = self.calibration_metrics.analyze_outcomes(confidences, outcomes)
        
        successes = sum(1 for r in self.run_summaries if r.get("success", False))
        
        # Compute aggregate latency metrics
        latency_metrics = self._compute_aggregate_latency()
        
        # Build comprehensive report
        report = {
            "summary": {
                "total_runs": len(self.run_summaries),
                "successful_runs": successes,
                "failed_runs": len(self.run_summaries) - successes,
                "success_rate": successes / len(self.run_summaries) if self.run_summaries else 0,
                "mean_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0,
            },
            "calibration": {
                "ece": calibration_result.ece,
                "mce": calibration_result.mce,
                "brier_score": calibration_result.brier_score,
                "spearman_rho": calibration_result.spearman_rho,
                "pearson_rho": calibration_result.pearson_rho,
                "auroc": calibration_result.auroc,
                "mean_accuracy": calibration_result.mean_accuracy,
                "bin_accuracies": calibration_result.bin_accuracies,
                "bin_confidences": calibration_result.bin_confidences,
                "bin_counts": calibration_result.bin_counts,
            },
            "latency": latency_metrics,
            "outcome_analysis": {
                "mean_confidence_success": outcome_analysis.mean_confidence_success,
                "mean_confidence_failure": outcome_analysis.mean_confidence_failure,
                "confidence_gap": outcome_analysis.confidence_gap,
                "spearman_rho": outcome_analysis.spearman_rho,
                "spearman_p_value": outcome_analysis.spearman_p_value,
                "auroc": outcome_analysis.auroc,
                "overconfidence_rate": outcome_analysis.overconfidence_rate,
                "underconfidence_rate": outcome_analysis.underconfidence_rate,
            },
            "interpretation": self._interpret_results(calibration_result, outcome_analysis),
            "runs": self.run_summaries,
        }
        
        output_path = os.path.join(self.config.output, "uncertainty_analysis.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("UNCERTAINTY ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total runs analyzed: {report['summary']['total_runs']}")
        print(f"Success rate: {report['summary']['success_rate']:.1%}")
        print(f"Overall mean confidence: {report['summary']['mean_confidence']:.3f}")
        print("-" * 40)
        print("CALIBRATION METRICS:")
        print(f"  ECE (Expected Calibration Error): {calibration_result.ece:.4f}")
        print(f"  Brier Score: {calibration_result.brier_score:.4f}")
        print(f"  Spearman rho (confidence-success): {calibration_result.spearman_rho:.4f}")
        print(f"  AUROC (discrimination): {calibration_result.auroc:.4f}")
        print("-" * 40)
        print("OUTCOME ANALYSIS:")
        print(f"  Mean confidence (success): {outcome_analysis.mean_confidence_success:.3f}")
        print(f"  Mean confidence (failure): {outcome_analysis.mean_confidence_failure:.3f}")
        print(f"  Confidence gap: {outcome_analysis.confidence_gap:+.3f}")
        print(f"  Overconfidence rate: {outcome_analysis.overconfidence_rate:.1%}")
        print(f"  Underconfidence rate: {outcome_analysis.underconfidence_rate:.1%}")
        print("-" * 40)
        print("LATENCY METRICS:")
        print(f"  Mean wall-clock time: {latency_metrics['mean_total_time']:.2f}s")
        print(f"  Mean step latency: {latency_metrics['mean_step_latency']:.2f}s")
        print(f"  P95 trajectory time: {latency_metrics['p95_total_time']:.2f}s")
        print(f"  P99 trajectory time: {latency_metrics['p99_total_time']:.2f}s")
        print("=" * 60 + "\n")
    
    def _interpret_results(
        self, 
        calibration: CalibrationResult, 
        outcome: OutcomeAnalysis
    ) -> Dict[str, str]:
        """Generate human-readable interpretations of the metrics."""
        interpretations = {}
        
        # ECE interpretation
        if calibration.ece < 0.05:
            interpretations["ece"] = "Well-calibrated: confidence closely matches actual success rates"
        elif calibration.ece < 0.15:
            interpretations["ece"] = "Moderately calibrated: some gap between confidence and success"
        else:
            interpretations["ece"] = "Poorly calibrated: large discrepancy between confidence and success"
        
        # Spearman interpretation
        if calibration.spearman_rho is not None:
            if calibration.spearman_rho > 0.3:
                interpretations["spearman"] = "Good: higher confidence correlates with success"
            elif calibration.spearman_rho > 0.1:
                interpretations["spearman"] = "Weak: slight correlation between confidence and success"
            elif calibration.spearman_rho > -0.1:
                interpretations["spearman"] = "No correlation: confidence doesn't predict success"
            else:
                interpretations["spearman"] = "Inverse: higher confidence correlates with FAILURE (problematic)"
        
        # AUROC interpretation
        if calibration.auroc is not None:
            if calibration.auroc > 0.7:
                interpretations["auroc"] = "Good discrimination: can distinguish success from failure"
            elif calibration.auroc > 0.55:
                interpretations["auroc"] = "Weak discrimination: limited ability to distinguish outcomes"
            else:
                interpretations["auroc"] = "Poor discrimination: confidence doesn't help predict outcomes"
        
        # Confidence gap interpretation
        if outcome.confidence_gap > 0.05:
            interpretations["confidence_gap"] = "Healthy: agent is more confident when successful"
        elif outcome.confidence_gap > -0.05:
            interpretations["confidence_gap"] = "Flat: similar confidence for success and failure"
        else:
            interpretations["confidence_gap"] = "Inverted: agent is MORE confident when failing (problematic)"
        
        # Overconfidence interpretation
        if outcome.overconfidence_rate > 0.5:
            interpretations["overconfidence"] = "High overconfidence: many failures with high confidence"
        elif outcome.overconfidence_rate > 0.2:
            interpretations["overconfidence"] = "Moderate overconfidence: some failures despite confidence"
        else:
            interpretations["overconfidence"] = "Low overconfidence: failures tend to have lower confidence"
        
        return interpretations
    
    def _compute_aggregate_latency(self) -> Dict[str, Any]:
        """Compute aggregate latency metrics across all runs."""
        if not self.run_summaries:
            return {
                "mean_total_time": 0.0,
                "mean_step_latency": 0.0,
                "p50_total_time": 0.0,
                "p95_total_time": 0.0,
                "p99_total_time": 0.0,
                "total_steps": 0,
            }
        
        # Collect trajectory times and step latencies
        total_times = []
        all_step_latencies = []
        
        for run in self.run_summaries:
            latency = run.get("latency", {})
            if latency:
                total_time = latency.get("total_time", 0.0)
                if total_time > 0:
                    total_times.append(total_time)
                step_latencies = latency.get("step_latencies", [])
                all_step_latencies.extend(step_latencies)
        
        if not total_times:
            return {
                "mean_total_time": 0.0,
                "mean_step_latency": 0.0,
                "p50_total_time": 0.0,
                "p95_total_time": 0.0,
                "p99_total_time": 0.0,
                "total_steps": 0,
            }
        
        # Sort for percentile calculation
        sorted_times = sorted(total_times)
        n = len(sorted_times)
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]
        
        return {
            "mean_total_time": sum(total_times) / n,
            "min_total_time": min(total_times),
            "max_total_time": max(total_times),
            "p50_total_time": percentile(sorted_times, 50),
            "p95_total_time": percentile(sorted_times, 95),
            "p99_total_time": percentile(sorted_times, 99),
            "mean_step_latency": sum(all_step_latencies) / len(all_step_latencies) if all_step_latencies else 0.0,
            "total_steps": len(all_step_latencies),
            "total_trajectories": n,
        }
    
    def record_completion(self, agent: str, task: str, index: SampleIndex, result: TaskOutput):
        def calculate_overall_worker():
            nonlocal agent, task, index, result
            task_client = self.tasks[task]
            overall = task_client.calculate_overall(self.completions[agent][task])
            with open(os.path.join(self.get_output_dir(agent, task), "overall.json"), "w") as f:
                f.write(json.dumps(overall, indent=4, ensure_ascii=False))
        
        overall_calculation = False
        with self.assignment_lock:
            if agent not in self.completions:
                self.completions[agent] = {}
            if task not in self.completions[agent]:
                self.completions[agent][task] = []
            result.index = index
            self.completions[agent][task].append(result)
            if len(self.completions[agent][task]) == len(self.task_indices[task]):
                overall_calculation = True
        if overall_calculation:
            output_dir = self.get_output_dir(agent, task)
            if os.path.exists(os.path.join(output_dir, "overall.json")):
                return
            threading.Thread(target=calculate_overall_worker).start()
    
    def finish_callback(
        self, agent: str, task: str, index: SampleIndex, result: TaskClientOutput
    ):
        # Get uncertainty summary from wrapped agent
        if agent in self.wrapped_agents and task in self.wrapped_agents[agent]:
            wrapper = self.wrapped_agents[agent][task]
            uncertainty_summary = wrapper.get_uncertainty_summary()
            wrapper.reset()  # Reset for next run
            
            # Determine task success (check for actual resolution, not just no error)
            task_success = result.error is None and result.output is not None
            if task_success and result.output:
                # For SWE-bench, check if patch was resolved
                if hasattr(result.output, 'resolved'):
                    task_success = result.output.resolved
                elif hasattr(result.output, 'success'):
                    task_success = result.output.success
                elif hasattr(result.output, 'score'):
                    task_success = result.output.score > 0
            
            # Record summary
            summary = {
                "agent": agent,
                "task": task,
                "index": index,
                "success": task_success,
                **uncertainty_summary,
            }
            self.run_summaries.append(summary)
            
            print(ColorMessage.cyan(
                f"Completed {agent}/{task}#{index}: confidence={uncertainty_summary.get('mean_confidence', 0):.3f}, "
                f"trend={uncertainty_summary.get('trend', 'stable')}, "
                f"success={task_success}"
            ))
        
        if result.error == TaskError.NOT_AVAILABLE.value:
            print(ColorMessage.yellow(f"Warning: {task} is not available, retrying."))
            with self.assignment_lock:
                self.remaining_tasks[agent][task].insert(0, index)
                self.free_worker.agent[agent] += 1
                self.free_worker.task[task] += 1
                self.running_count -= 1
            return
        
        if result.error is not None:
            print(ColorMessage.yellow(
                f"Warning: {agent}/{task}#{index} failed with error {result.error} {result.info} {result.output}"
            ))
            if self.auto_retry:
                with self.assignment_lock:
                    self.remaining_tasks[agent][task].insert(0, index)
        
        output_folder = self.get_output_dir(agent, task)
        os.makedirs(output_folder, exist_ok=True)
        timestamp: int = int(time.time() * 1000)
        time_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        write_to_file = (
            json.dumps({
                "index": index,
                **result.dict(),
                "time": {"timestamp": timestamp, "str": time_str},
            })
            + "\n"
        )
        if not result.error:
            target_file = os.path.join(output_folder, "runs.jsonl")
            with self.assignment_lock:
                self.finished_count += 1
            self.record_completion(agent, task, index, result.output)
            self.overall_tqdm.update(1)
            self.tqdm_ordered_by_agent[agent].update(1)
        else:
            target_file = os.path.join(output_folder, "error.jsonl")
        with open(target_file, "a+", encoding="utf-8") as f:
            f.write(write_to_file)
        
        with self.assignment_lock:
            self.free_worker.agent[agent] += 1
            self.free_worker.task[task] += 1
            self.running_count -= 1
    
    def start_worker(
        self,
        agent: str,
        task: str,
        index: SampleIndex,
        finish_callback: Union[Callable[[str, str, SampleIndex, TaskClientOutput], None], None] = None,
    ):
        def worker_thread():
            nonlocal agent, task, index, finish_callback
            
            # Use wrapped agent for this task
            wrapped = self.wrapped_agents.get(agent, {}).get(task)
            agent_to_use = wrapped if wrapped else self.agents[agent]
            
            result = self.tasks[task].run_sample(index, agent_to_use)
            
            if finish_callback:
                finish_callback(agent, task, index, result)
        
        with self.assignment_lock:
            self.running_count += 1
        threading.Thread(target=worker_thread).start()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AgentBench with uncertainty tracking")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/assignments/default.yaml",
        help="Path to assignment config file"
    )
    parser.add_argument(
        "--auto-retry", "-r", action="store_true", dest="retry",
        help="Automatically retry failed samples"
    )
    args = parser.parse_args()
    
    loader = ConfigLoader()
    config_ = loader.load_from(args.config)
    value = AssignmentConfig.parse_obj(config_)
    value = AssignmentConfig.post_validate(value)
    
    with std_out_err_redirect_tqdm() as orig_stdout:
        UncertaintyAssigner(value, args.retry).start(tqdm_out=orig_stdout)
