"""
Uncertainty-Aware Assigner for AgentBench FC (Function Calling).

This is a modified version of the Assigner that wraps agents with
uncertainty tracking, capturing real-time confidence signals during
benchmark runs.

Supported Tasks (AgentBench FC):
    - alfworld-std (AF): Household tasks
    - dbbench-std (DB): Database SQL tasks
    - os-std (OS): OS interaction tasks
    - kg-std (KG): Knowledge graph QA
    - webshop-std (WS): Web shopping

Usage:
    python -m src.uncertainty_assigner --config configs/assignments/agentbench-fc.yaml

The uncertainty data is saved alongside the regular run results.
"""

import datetime
import json
import os
import random
import threading
import time
from typing import Dict, List, Union
from typing import Tuple, Callable, Iterator
import contextlib
import sys
from tqdm.contrib import DummyTqdmFile

import yaml
from tqdm import tqdm

from src.client.task import TaskError
from src.client import TaskClient, AgentClient
from src.configs import ConfigLoader
from src.typings import AssignmentConfig, SampleIndex, TaskOutput, TaskClientOutput
from src.utils import ColorMessage
from src.utils import Graph, MaxFlow

# Import uncertainty components
from src.agentbench_debug.uncertainty import (
    UncertaintyTracker,
    ConfidenceExtractor,
    OrchestrationHarness,
    EvaluationConfig,
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
    Wrapper that intercepts agent inference calls to track uncertainty.
    """
    
    def __init__(self, agent: AgentClient, task_name: str):
        self._agent = agent
        self._task_name = task_name
        self._tracker = UncertaintyTracker()
        self._extractor = ConfidenceExtractor()
        self._step_count = 0
        self._last_raw_response = None
        
    def inference(self, history) -> str:
        """Intercept inference to extract confidence."""
        self._step_count += 1
        
        # Call original agent
        response = self._agent.inference(history)
        
        # Extract confidence from the response
        confidence = self._extract_confidence(response)
        
        # Infer action type from response content
        action_type = self._infer_action_type(response)
        
        # Record in tracker
        self._tracker.record_response(
            content=response,
            action_name=f"step_{self._step_count}",
            action_type=action_type,
        )
        
        return response
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence from response text."""
        # Try to get raw response if agent stored it
        raw_response = getattr(self._agent, '_last_raw_response', None)
        
        if raw_response:
            signals = self._extractor.extract(raw_response, api_type="auto", content=response)
        else:
            signals = self._extractor.extract(response, api_type="generic")
        
        return signals.confidence
    
    def _infer_action_type(self, content: str) -> str:
        """Infer action type from response content for AgentBench FC tasks."""
        content_lower = content.lower() if content else ""
        
        # AgentBench FC task-specific actions
        # ALFWorld - household tasks
        if any(kw in content_lower for kw in ["take_action", "go to", "pick up", "put", "open", "close", "toggle", "heat", "cool", "clean"]):
            return "environment_action"
        
        # DBBench - SQL queries
        if any(kw in content_lower for kw in ["execute_sql", "select", "insert", "update", "delete from", "create table"]):
            return "query"
        if "commit_final_answer" in content_lower:
            return "submit"
        
        # OS Interaction - bash commands
        if any(kw in content_lower for kw in ["bash_action", "cd ", "ls ", "cat ", "grep ", "chmod", "sudo", "apt"]):
            return "shell_command"
        if any(kw in content_lower for kw in ["finish_action", "answer_action"]):
            return "submit"
        
        # WebShop - e-commerce navigation
        if "search_action" in content_lower or "search for" in content_lower:
            return "search"
        if "click_action" in content_lower or "click on" in content_lower:
            return "navigation"
        if "buy" in content_lower or "add to cart" in content_lower:
            return "purchase"
        
        # Generic patterns
        if any(kw in content_lower for kw in ["error", "unable", "cannot", "failed"]):
            return "error_handling"
        if any(kw in content_lower for kw in ["clarify", "confirm", "please provide"]):
            return "clarify"
        if any(kw in content_lower for kw in ["search", "find", "look", "query"]):
            return "query"
        if any(kw in content_lower for kw in ["delete", "remove"]):
            return "delete"
        if any(kw in content_lower for kw in ["create", "write", "save"]):
            return "write"
        
        return "respond"
    
    def get_uncertainty_analysis(self) -> Dict:
        """Get uncertainty analysis for this run."""
        return self._tracker.get_analysis()
    
    def get_confidence_history(self) -> List[float]:
        """Get confidence values for all steps."""
        return self._tracker.get_confidence_history()
    
    def reset(self):
        """Reset for a new run."""
        self._tracker.reset()
        self._step_count = 0
    
    def __getattr__(self, name):
        """Proxy other attributes to wrapped agent."""
        return getattr(self._agent, name)


class UncertaintyAssigner:
    """
    Modified Assigner that tracks uncertainty during benchmark runs.
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
        self.uncertainty_results: Dict[str, Dict[str, List[Dict]]] = {}  # {agent: {task: [results]}}
        
        # Check/create output folder
        if not os.path.exists(self.config.output):
            os.makedirs(self.config.output)
            with open(os.path.join(self.config.output, "config.yaml"), "w") as f:
                f.write(yaml.dump(self.config.dict()))

        # Walk through existing results
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

        count = sum(
            len(self.remaining_tasks[agent][task])
            for agent in self.remaining_tasks
            for task in self.remaining_tasks[agent]
        )
        print(ColorMessage.cyan(f"Message: {count} samples remaining."))

        for agent in self.remaining_tasks:
            samples = sum(len(self.remaining_tasks[agent][task]) for task in self.remaining_tasks[agent])
            if samples == 0:
                continue
            print(ColorMessage.cyan(f"Agent {agent} needs to run {len(self.remaining_tasks[agent])} tasks with {samples} samples"))

        # Create agents with uncertainty wrappers
        for agent in self.remaining_tasks:
            print(ColorMessage.green(f"Creating uncertainty-aware agent: {agent}"))
            self.agents[agent] = self.config.definition.agent[agent].create()
            self.wrapped_agents[agent] = {}
            self.uncertainty_results[agent] = {}
            
            for task in self.remaining_tasks[agent]:
                # Create a wrapper for each agent-task pair
                self.wrapped_agents[agent][task] = UncertaintyAwareAgentWrapper(
                    self.agents[agent], task
                )
                self.uncertainty_results[agent][task] = []

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
        self.started_count = sum(
            len(self.remaining_tasks[agent][task])
            for agent in self.remaining_tasks
            for task in self.remaining_tasks[agent]
        )
        generator = self.worker_generator()
        self.overall_tqdm = tqdm(total=self.started_count, desc="Total", position=0, file=tqdm_out)
        
        for idx, agent in enumerate(self.remaining_tasks.keys()):
            self.tqdm_ordered_by_agent[agent] = tqdm(
                total=sum(len(self.remaining_tasks[agent][task]) for task in self.remaining_tasks[agent]),
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

        # Save uncertainty analysis
        self._save_uncertainty_results()

        final_message = (
            "\n\n============================================\n"
            + ColorMessage.cyan(f"Message: {self.started_count} sample(s) started. ")
            + "\n"
            + ColorMessage.green(f"   >> {self.finished_count} sample(s) finished successfully.")
            + "\n"
        )
        if self.started_count != self.finished_count:
            final_message += ColorMessage.red(
                f"   >> {self.started_count - self.finished_count} sample(s) failed."
            ) + "\n"
        final_message += ColorMessage.cyan(f"   >> results are saved to {self.config.output}") + "\n"
        final_message += ColorMessage.cyan(f"   >> uncertainty analysis saved to {self.config.output}/uncertainty_analysis.json") + "\n"
        final_message += "============================================\n\n"
        print(final_message)

    def _save_uncertainty_results(self):
        """Save uncertainty analysis to file."""
        # Aggregate results
        aggregate = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_runs": 0,
            "by_agent_task": {},
            "overall_metrics": {
                "mean_confidence": 0,
                "total_steps": 0,
                "high_uncertainty_runs": 0,
            }
        }
        
        all_confidences = []
        
        for agent in self.uncertainty_results:
            aggregate["by_agent_task"][agent] = {}
            for task in self.uncertainty_results[agent]:
                results = self.uncertainty_results[agent][task]
                if not results:
                    continue
                
                aggregate["total_runs"] += len(results)
                
                task_confidences = []
                for r in results:
                    task_confidences.append(r.get("mean_confidence", 0.7))
                    all_confidences.append(r.get("mean_confidence", 0.7))
                
                aggregate["by_agent_task"][agent][task] = {
                    "runs": len(results),
                    "mean_confidence": sum(task_confidences) / len(task_confidences) if task_confidences else 0,
                    "results": results,
                }
        
        if all_confidences:
            aggregate["overall_metrics"]["mean_confidence"] = sum(all_confidences) / len(all_confidences)
        
        # Save to file
        output_file = os.path.join(self.config.output, "uncertainty_analysis.json")
        with open(output_file, "w") as f:
            json.dump(aggregate, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("UNCERTAINTY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total runs analyzed: {aggregate['total_runs']}")
        print(f"Overall mean confidence: {aggregate['overall_metrics']['mean_confidence']:.3f}")
        print("="*60 + "\n")

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

    def finish_callback(self, agent: str, task: str, index: SampleIndex, result: TaskClientOutput):
        # Get uncertainty analysis from wrapper
        wrapper = self.wrapped_agents[agent][task]
        uncertainty_analysis = wrapper.get_uncertainty_analysis()
        confidence_history = wrapper.get_confidence_history()
        
        # Store uncertainty result
        uncertainty_result = {
            "index": index,
            "success": result.error is None and result.output is not None,
            "n_steps": uncertainty_analysis.get("n_steps", 0),
            "mean_confidence": uncertainty_analysis.get("mean_confidence", 0.7),
            "min_confidence": uncertainty_analysis.get("min_confidence", 0.7),
            "trajectory_uncertainty": uncertainty_analysis.get("trajectory_uncertainty", 0),
            "trend": uncertainty_analysis.get("trend", "stable"),
            "confidence_history": confidence_history,
        }
        
        with self.assignment_lock:
            self.uncertainty_results[agent][task].append(uncertainty_result)
        
        # Reset wrapper for next run
        wrapper.reset()
        
        # Handle errors
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
                f"Warning: {agent}/{task}#{index} failed with error {result.error} "
                f"(confidence: {uncertainty_result['mean_confidence']:.3f})"
            ))
            if self.auto_retry:
                with self.assignment_lock:
                    self.remaining_tasks[agent][task].insert(0, index)

        output_folder = self.get_output_dir(agent, task)
        os.makedirs(output_folder, exist_ok=True)
        timestamp: int = int(time.time() * 1000)
        time_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        
        # Add uncertainty data to the saved result
        write_data = {
            "index": index,
            **result.dict(),
            "uncertainty": uncertainty_result,
            "time": {"timestamp": timestamp, "str": time_str},
        }
        write_to_file = json.dumps(write_data) + "\n"
        
        if not result.error:
            target_file = os.path.join(output_folder, "runs.jsonl")
            with self.assignment_lock:
                self.finished_count += 1
            self.record_completion(agent, task, index, result.output)
            self.overall_tqdm.update(1)
            self.tqdm_ordered_by_agent[agent].update(1)
            
            # Print confidence info
            print(ColorMessage.cyan(
                f"Completed {agent}/{task}#{index}: confidence={uncertainty_result['mean_confidence']:.3f}, "
                f"trend={uncertainty_result['trend']}"
            ))
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
            
            # Use the wrapped agent instead of the raw agent
            wrapped_agent = self.wrapped_agents[agent][task]
            result = self.tasks[task].run_sample(index, wrapped_agent)
            
            if finish_callback:
                finish_callback(agent, task, index, result)

        with self.assignment_lock:
            self.running_count += 1
        threading.Thread(target=worker_thread).start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AgentBench with uncertainty tracking")
    parser.add_argument("--config", "-c", type=str, default="configs/assignments/default.yaml")
    parser.add_argument("--auto-retry", "-r", action="store_true", dest="retry")
    args = parser.parse_args()

    loader = ConfigLoader()
    config_ = loader.load_from(args.config)
    value = AssignmentConfig.parse_obj(config_)
    value = AssignmentConfig.post_validate(value)
    
    print("\n" + "="*60)
    print("UNCERTAINTY-AWARE AGENTBENCH EVALUATION")
    print("="*60)
    print("This run will track agent confidence in real-time.")
    print("="*60 + "\n")
    
    with std_out_err_redirect_tqdm() as orig_stdout:
        UncertaintyAssigner(value, args.retry).start(tqdm_out=orig_stdout)

