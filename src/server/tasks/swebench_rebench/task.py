import asyncio
import json
import logging
import os
import weakref
from typing import List, Dict, Any, Optional
from pathlib import Path

from agentrl.worker.environment import create_controller
from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (
    AgentCancelledException,
    SampleIndex,
    SampleStatus,
    TaskSampleExecutionResult,
    TaskOutput,
    RewardHistoryItem
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam
)

from .environment import SWEBenchRebenchEnvironmentDelegation
from .container import SWEBenchContainer


SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing bugs in open-source repositories.

You will be given:
1. A problem statement describing a bug or issue
2. Access to a Docker environment with the repository already set up at /testbed
3. Tools to explore the code, run commands, and submit your fix

Your goal is to:
1. Understand the problem by reading the issue description
2. Explore the repository structure and locate relevant code
3. Identify the root cause of the bug
4. Develop and test a fix
5. Submit a patch in unified diff format

Available tools:
- bash_command: Execute bash commands (git, grep, find, pytest, etc.)
- read_file: Read file contents
- search_code: Search for patterns in Python files
- submit_patch: Submit your final patch (ends the task)

Work methodically:
1. First explore the repository structure
2. Locate the relevant files mentioned in the issue
3. Understand the existing code
4. Identify the bug
5. Create a fix
6. Test your fix if possible
7. Generate a unified diff patch
8. Submit the patch

Important:
- The repository is at /testbed
- You can run tests, git commands, etc.
- Your patch should be in unified diff format (git diff output)
- Only submit when you're confident in your fix
- If you need to see test results, run them before submitting
"""


class SWEBenchRebenchTask(Task):
    """
    SWE-rebench task implementation.

    Provides interactive environment for agents to explore codebases
    and generate patches for software engineering bugs.
    """

    def __init__(
        self,
        data_file: str,
        output_file: str = "predictions.jsonl",
        max_round: int = 20,
        command_timeout: int = 60,
        env_driver: str = 'docker',
        env_options: Optional[dict] = None,
        docker_config: Optional[dict] = None,
        **configs
    ):
        super().__init__(**configs)
        self.full_async = True
        self.logger = logging.getLogger(__name__)

        self.data_file = data_file
        self.output_file = output_file
        self.max_round = max_round
        self.command_timeout = command_timeout
        self.docker_config = docker_config or {}

        # Load dataset
        self.dataset = []
        self.logger.info(f"Loading dataset from: {data_file}")
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))

        self.logger.info(f"Loaded {len(self.dataset)} instances from SWE-rebench")

        # Initialize environment controller
        self.env_delegation = SWEBenchRebenchEnvironmentDelegation()
        self.env_controller = create_controller(env_driver, self.env_delegation, **env_options or {})
        self.env_controller_background_task = None

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        # Clear output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.dataset)))

    async def start_sample(self, index: int, session: Session) -> TaskSampleExecutionResult:
        """Execute one SWE-bench instance."""

        # Start environment controller if needed
        if not self.env_controller_background_task:
            self.env_controller_background_task = asyncio.create_task(
                self.env_controller.background_task()
            )
            weakref.finalize(self, self.env_controller_background_task.cancel)

        instance = self.dataset[index]
        instance_id = instance['instance_id']
        docker_image = instance.get('docker_image')

        # Skip instances without pre-built Docker images
        if not docker_image or docker_image == "None":
            self.logger.warning(f"Skipping instance {instance_id}: No Docker image available")
            self._save_prediction(instance_id, "")  # Save empty patch
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(
                status=SampleStatus.COMPLETED,
                result={"instance_id": instance_id, "skipped": True, "reason": "no_docker_image"}
            )

        self.logger.info(f"Starting instance: {instance_id}")
        self.logger.info(f"Docker image: {docker_image}")

        # Pull Docker image if needed
        try:
            await self._ensure_image_pulled(docker_image)
        except Exception as e:
            self.logger.error(f"Failed to pull image {docker_image}: {e}")
            self._save_prediction(instance_id, "")  # Save empty patch
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_ERROR,
                result={"instance_id": instance_id, "error": f"Failed to pull Docker image: {e}"}
            )

        # Create container with subtype 'swerebench' and pass the full image name
        container = SWEBenchContainer(self.env_controller, 'swerebench', docker_image)

        try:
            await container.initialize()
            result = await self._execute_instance(session, instance, container)
            return result

        except AgentCancelledException:
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
        except Exception as e:
            self.logger.exception(f"Error in instance {instance_id}")
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_ERROR,
                result={"instance_id": instance_id, "error": str(e)}
            )
        finally:
            try:
                await container.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up container: {e}")

    async def _ensure_image_pulled(self, image: str):
        """Ensure Docker image is pulled."""
        # Use docker command via controller to check/pull image
        # This is a simplified approach - in production might use Docker SDK
        self.logger.info(f"Ensuring image is available: {image}")
        # The environment controller will handle pulling when container is created
        pass

    async def _execute_instance(
        self,
        session: Session,
        instance: dict,
        container: SWEBenchContainer
    ) -> TaskSampleExecutionResult:
        """Execute agent interaction for one instance."""

        instance_id = instance['instance_id']

        # Inject system prompt
        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=SYSTEM_PROMPT
        ))

        # Inject problem statement
        problem_prompt = self._build_problem_prompt(instance)
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=problem_prompt
        ))

        # Interaction loop
        submitted_patch = None

        for round_num in range(self.max_round):
            self.logger.info(f"Round {round_num + 1}/{self.max_round} for {instance_id}")

            # Get agent action
            response = await session.action()

            # Extract tool calls
            tool_calls = []
            for message in response.messages:
                tool_calls.extend(message.get('tool_calls', []) or [])

            if not tool_calls:
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content='No tool calls found. Please use one of the available tools.'
                ))
                continue

            # Process first tool call
            tool_call = tool_calls[0]
            call_id = tool_call.get('id', '')

            try:
                function_name = tool_call.get('function', {}).get('name', '')
                arguments = tool_call.get('function', {}).get('arguments', '{}')
                arguments = json.loads(arguments)
            except Exception as e:
                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content=f'Error parsing tool call: {e}'
                ))
                continue

            # Execute tool
            if function_name == 'bash_command':
                command = arguments.get('command', '')
                output = await container.execute_bash(command, self.command_timeout)

                # Truncate long output
                if len(output) > 800:
                    output = output[:780] + "\n[truncated - output too long]"

                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content=output if output else "(no output)"
                ))

            elif function_name == 'read_file':
                file_path = arguments.get('file_path', '')
                output = await container.read_file(file_path)

                if len(output) > 2000:
                    output = output[:2000] + "\n[truncated - file too long]"

                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content=output
                ))

            elif function_name == 'search_code':
                pattern = arguments.get('pattern', '')
                directory = arguments.get('directory', '.')
                output = await container.search_code(pattern, directory)

                if len(output) > 1000:
                    output = output[:1000] + "\n[truncated - too many results]"

                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content=output
                ))

            elif function_name == 'submit_patch':
                patch = arguments.get('patch', '')
                submitted_patch = patch

                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content='Patch submitted successfully.'
                ))

                # Save prediction
                self._save_prediction(instance_id, patch)

                # End episode
                session.inject(RewardHistoryItem(reward=0, score=0))  # No in-task evaluation
                return TaskSampleExecutionResult(
                    status=SampleStatus.COMPLETED,
                    result={
                        "instance_id": instance_id,
                        "patch_submitted": True,
                        "patch_length": len(patch)
                    }
                )

            else:
                session.inject(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=call_id,
                    content=f'Unknown tool: {function_name}'
                ))

        # Max rounds reached
        self.logger.warning(f"Max rounds reached for {instance_id}")

        # Save empty patch if none submitted
        if submitted_patch is None:
            self._save_prediction(instance_id, "")

        session.inject(RewardHistoryItem(reward=0, score=0))
        return TaskSampleExecutionResult(
            status=SampleStatus.TASK_LIMIT_REACHED,
            result={
                "instance_id": instance_id,
                "patch_submitted": submitted_patch is not None
            }
        )

    def _build_problem_prompt(self, instance: dict) -> str:
        """Build problem statement prompt."""
        prompt = f"""# Problem Statement

**Repository**: {instance['repo']}
**Base Commit**: {instance['base_commit']}

**Issue Description**:
{instance['problem_statement']}
"""

        # Add hints if available
        if instance.get('hints_text'):
            prompt += f"\n**Hints**:\n{instance['hints_text']}\n"

        # Add test information
        if instance.get('FAIL_TO_PASS'):
            prompt += f"\n**Tests that should pass after fix**:\n"
            for test in instance['FAIL_TO_PASS'][:5]:
                prompt += f"- {test}\n"

        prompt += "\n**Instructions**: The repository is available at /testbed. Explore the code, identify the bug, and submit a patch to fix it."

        return prompt

    def _save_prediction(self, instance_id: str, patch: str):
        """Save prediction to output file."""
        prediction = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": "agentbench"
        }

        with open(self.output_file, 'a') as f:
            f.write(json.dumps(prediction) + '\n')

        self.logger.info(f"Saved prediction for {instance_id}")

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """Calculate overall metrics."""
        total = len(results)
        completed = sum(1 for r in results if r.status == SampleStatus.COMPLETED)
        timeout = sum(1 for r in results if r.status == SampleStatus.TASK_LIMIT_REACHED)
        errors = sum(1 for r in results if r.status == SampleStatus.TASK_ERROR)

        with_patch = sum(
            1 for r in results
            if r.result and r.result.get('patch_submitted', False)
        )

        return {
            "total": total,
            "completed": completed,
            "timeout": timeout,
            "errors": errors,
            "patches_generated": with_patch,
            "completion_rate": completed / total if total > 0 else 0,
            "patch_rate": with_patch / total if total > 0 else 0
        }

    def release(self):
        """Cleanup on task completion."""
        self.logger.info(f"Task completed. Predictions saved to: {self.output_file}")
