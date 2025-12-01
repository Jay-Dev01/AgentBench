import enum

import requests

from src.typings import *
from src.utils import *
from .agent import AgentClient


class TaskError(enum.Enum):
    START_FAILED = "START_FAILED"
    INTERACT_FAILED = "INTERACT_FAILED"
    AGENT_FAILED = "AGENT_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class TaskClient:
    def __init__(
        self, name: str, controller_address: str = "http://localhost:5000/api", *_, **__,
    ) -> None:
        self.name = name
        self.controller_address = controller_address
        print("TaskClient created: {} ({})".format(name, controller_address))

    def get_indices(self) -> List[SampleIndex]:
        result = requests.get(
            self.controller_address + "/get_indices", params={"name": self.name}
        )
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        return result.json()

    def get_concurrency(self) -> int:
        try:
            result = requests.get(
                self.controller_address + "/list_workers"
            )
        except Exception as e:
            print(ColorMessage.yellow(f"Warning task {self.name} cannot connect to controller {e}"))
            return 0
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        result = result.json()
        if self.name not in result:
            print(ColorMessage.yellow(f"task {self.name} not found in worker list"))
            return 0
        concurrency = 0
        for worker in result[self.name]["workers"].values():
            status = worker.get("status")
            is_alive = False
            # Controller may return enum value (0) or string like "ALIVE"
            try:
                is_alive = (status == WorkerStatus.ALIVE) or (
                    isinstance(status, str) and status.upper() == "ALIVE"
                ) or (isinstance(status, int) and status == int(WorkerStatus.ALIVE))
            except Exception:
                is_alive = False
            if is_alive:
                concurrency += worker["capacity"] - worker["current"]
        return concurrency

    def run_sample(self, index: SampleIndex, agent: AgentClient) -> TaskClientOutput:
        try:
            result = requests.post(
                self.controller_address + "/start_sample",
                json=StartSampleRequest(name=self.name, index=index).dict(),
            )
        except Exception as e:
            return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e))
        if result.status_code == 406:
            return TaskClientOutput(
                error=TaskError.NOT_AVAILABLE.value, info=result.text
            )
        if result.status_code != 200:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value, info=result.text
            )
        result = result.json()
        # Accept legacy or FC-style responses. Extract session_id and output payload robustly.
        sid = None
        output_payload = None
        if isinstance(result, dict):
            # Session id might be at different locations
            if "session_id" in result:
                sid = result["session_id"]
            elif isinstance(result.get("session"), dict) and "id" in result["session"]:
                sid = result["session"]["id"]
            # Output payload may be nested or top-level (messages/tools)
            if "output" in result:
                output_payload = result["output"]
            elif ("messages" in result) or ("tools" in result):
                output_payload = {
                    "messages": result.get("messages", []),
                    "tools": result.get("tools", []),
                    "status": SampleStatus.RUNNING,
                }
        # Fallback: try to fetch latest session id from controller if missing
        if sid is None:
            try:
                probe = requests.get(self.controller_address + "/get_sessions", params={"name": self.name})
                if probe.status_code == 200:
                    sessions = probe.json()
                    # sessions could be a list of dicts with 'id'; choose the max id
                    if isinstance(sessions, list):
                        candidates = [s.get("id") for s in sessions if isinstance(s, dict) and "id" in s]
                        if candidates:
                            sid = max(candidates)
                    elif isinstance(sessions, dict):
                        # maybe { "sessions": [ {id:...}, ... ] }
                        items = sessions.get("sessions")
                        if isinstance(items, list):
                            candidates = [s.get("id") for s in items if isinstance(s, dict) and "id" in s]
                            if candidates:
                                sid = max(candidates)
            except Exception:
                pass
        if sid is None or output_payload is None:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=f"Invalid start_sample response: {result}",
            )
        # Normalize internal result
        result = {"session_id": sid, "output": output_payload}
        sid = result["session_id"]
        latest_result = result
        while SampleStatus(result["output"].get("status", SampleStatus.RUNNING)) == SampleStatus.RUNNING:
            try:
                output_payload = result["output"]
                response: AgentOutput
                # FC-style: controller provides messages/tools; agent must return assistant message with tool_calls
                if "messages" in output_payload and "tools" in output_payload:
                    # Try to call FC interface if available
                    if hasattr(agent, "inference_with_tools"):
                        assistant_message = getattr(agent, "inference_with_tools")(
                            messages=output_payload["messages"], tools=output_payload.get("tools", [])
                        )
                        response = AgentOutput(content={"messages": [assistant_message]})
                    else:
                        raise Exception("Agent does not support function-calling (tools) interface")
                else:
                    # Legacy: use history of role/content
                    content = agent.inference(output_payload["history"])
                    response = AgentOutput(content=content)
            except AgentContextLimitException:
                response = AgentOutput(status=AgentOutputStatus.AGENT_CONTEXT_LIMIT)
            except Exception as e:
                if hasattr(agent, "model_name"):
                    model_name = agent.model_name
                elif hasattr(agent, "name"):
                    model_name = agent.name
                else:
                    model_name = agent.__class__.__name__
                print(f"ERROR: {model_name}/{self.name} agent error", e)
                requests.post(
                    self.controller_address + "/cancel",
                    json=CancelRequest(session_id=sid).dict(),
                )
                return TaskClientOutput(
                    error=TaskError.AGENT_FAILED.value,
                    info=str(e),
                    output=latest_result,
                )

            try:
                result = requests.post(
                    self.controller_address + "/interact",
                    json=InteractRequest(
                        session_id=sid,
                        agent_response=response,
                    ).dict(),
                )
            except Exception as e:
                return TaskClientOutput(
                    error=TaskError.NETWORK_ERROR.value,
                    info=str(e),
                    output=latest_result,
                )
            if result.status_code != 200:
                requests.post(
                    self.controller_address + "/cancel",
                    json=CancelRequest(session_id=sid).dict(),
                )
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=result.text,
                    output=latest_result,
                )

            result = result.json()
            latest_result = result
        # TODO: check this type and check where history is
        return TaskClientOutput(output=result["output"])

    def calculate_overall(self, results: List[TaskOutput]) -> JSONSerializable:
        statistics = {s: 0 for s in SampleStatus}
        for result in results:
            statistics[SampleStatus(result.status)] += 1
        for s in SampleStatus:
            statistics[s] /= len(results)
        statistics["average_history_length"] = sum(
            [len(result.history) for result in results]
        ) / len(results)
        statistics["max_history_length"] = max(
            [len(result.history) for result in results]
        )
        statistics["min_history_length"] = min(
            [len(result.history) for result in results]
        )
        ret = {
            "total": len(results),
            "validation": statistics,
        }
        res = requests.post(
            self.controller_address + "/calculate_overall",
            json=CalculateOverallRequest(name=self.name, results=results).dict(),
        )
        if res.status_code != 200:
            raise TaskNetworkException(res.text)
        ret["custom"] = res.json()
        return ret
