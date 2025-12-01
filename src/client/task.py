import enum
import time
import json

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
            ex = TaskClientException(
                "start_failed_http",
                f"{result.status_code}: {result.text}",
            )
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=str(ex),
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
                    "tools": result.get("tools") or [],
                    "status": SampleStatus.RUNNING,
                }

        def _select_sid(items: List[dict]) -> Union[int, None]:
            # Prefer exact task + index match, then task-only, else max id
            def _sid(x): return x.get("id") if isinstance(x, dict) else None
            def _task_name(x):
                if not isinstance(x, dict):
                    return None
                return x.get("task") or x.get("name") or x.get("task_name")
            def _index(x): return x.get("index") if isinstance(x, dict) else None
            candidates = [_sid(s) for s in items if _task_name(s) == self.name and _index(s) == index and _sid(s) is not None]
            if candidates:
                return max(candidates)
            candidates = [_sid(s) for s in items if _task_name(s) == self.name and _sid(s) is not None]
            if candidates:
                return max(candidates)
            candidates = [_sid(s) for s in items if _sid(s) is not None]
            if candidates:
                return max(candidates)
            return None
        # Fallback: try to fetch latest session id from controller if missing
        if sid is None:
            try:
                # Use documented endpoint to enumerate sessions
                probe = requests.get(self.controller_address + "/list_sessions")
                if probe.status_code == 200:
                    sessions = probe.json()
                    items = None
                    if isinstance(sessions, dict) and isinstance(sessions.get("sessions"), list):
                        items = sessions.get("sessions")
                    elif isinstance(sessions, list):
                        items = sessions
                    elif isinstance(sessions, dict):
                        # dict-of-dicts keyed by id -> details
                        try:
                            items = [{"id": int(k), **v} for k, v in sessions.items() if isinstance(v, dict)]
                        except Exception:
                            items = [{"id": k, **v} for k, v in sessions.items() if isinstance(v, dict)]
                    if isinstance(items, list):
                        candidate = _select_sid(items)
                        if candidate is not None:
                            sid = candidate
            except Exception:
                pass
        # Legacy fallback: some controllers expose /get_sessions?name=<task>
        if sid is None:
            try:
                probe = requests.get(self.controller_address + "/get_sessions", params={"name": self.name})
                if probe.status_code == 200:
                    sessions = probe.json()
                    # sessions could be a simple list of dicts
                    if isinstance(sessions, list):
                        candidates = [s.get("id") for s in sessions if isinstance(s, dict) and s.get("id") is not None]
                        if candidates:
                            sid = max(candidates)
                    elif isinstance(sessions, dict):
                        items = sessions.get("sessions")
                        if isinstance(items, list):
                            candidates = [s.get("id") for s in items if isinstance(s, dict) and s.get("id") is not None]
                            if candidates:
                                sid = max(candidates)
            except Exception:
                pass
        # If still missing, poll briefly to allow controller to register the session (race-safe)
        if sid is None:
            for _ in range(10):
                try:
                    probe = requests.get(self.controller_address + "/list_sessions")
                    if probe.status_code == 200:
                        sessions = probe.json()
                        items = None
                        if isinstance(sessions, dict) and isinstance(sessions.get("sessions"), list):
                            items = sessions.get("sessions")
                        elif isinstance(sessions, list):
                            items = sessions
                        elif isinstance(sessions, dict):
                            try:
                                items = [{"id": int(k), **v} for k, v in sessions.items() if isinstance(v, dict)]
                            except Exception:
                                items = [{"id": k, **v} for k, v in sessions.items() if isinstance(v, dict)]
                        if isinstance(items, list):
                            candidate = _select_sid(items)
                            if candidate is not None:
                                sid = candidate
                                break
                except Exception:
                    pass
                time.sleep(0.3)
        # Final fallback: poll legacy /get_sessions for this task name
        if sid is None:
            for _ in range(10):
                try:
                    probe = requests.get(self.controller_address + "/get_sessions", params={"name": self.name})
                    if probe.status_code == 200:
                        sessions = probe.json()
                        if isinstance(sessions, list):
                            ids = [s.get("id") for s in sessions if isinstance(s, dict) and s.get("id") is not None]
                            if ids:
                                sid = max(ids)
                                break
                        elif isinstance(sessions, dict) and isinstance(sessions.get("sessions"), list):
                            items = sessions.get("sessions")
                            ids = [s.get("id") for s in items if isinstance(s, dict) and s.get("id") is not None]
                            if ids:
                                sid = max(ids)
                                break
                except Exception:
                    pass
                time.sleep(0.3)
        if sid is None or output_payload is None:
            # Build a detailed diagnostic with raw payload (truncated)
            try:
                raw_str = json.dumps(result, ensure_ascii=False)
            except Exception:
                raw_str = str(result)
            # Try to gather session lists for deeper debugging
            debug_list = None
            debug_get = None
            try:
                r1 = requests.get(self.controller_address + "/list_sessions")
                if r1.status_code == 200:
                    debug_list = r1.json()
            except Exception:
                pass
            try:
                r2 = requests.get(self.controller_address + "/get_sessions", params={"name": self.name})
                if r2.status_code == 200:
                    debug_get = r2.json()
            except Exception:
                pass
            diag = {
                "has_session_id": "session_id" in result,
                "has_session_obj": isinstance(result.get("session"), dict),
                "has_output": "output" in result,
                "has_messages": "messages" in result,
                "tools_type": type(result.get("tools")).__name__,
                "list_sessions": debug_list,
                "get_sessions": debug_get,
            }
            ex = TaskClientException(
                "start_failed_invalid_response",
                f"diag={diag}, raw={raw_str[:2000]}",
            )
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=str(ex),
            )
        # Normalize internal result
        result = {"session_id": sid, "output": output_payload}
        sid = result["session_id"]

        # Pre-interact confirmation: ensure the chosen sid is visible and the best match
        def _confirm_sid(current_sid: Union[int, None]) -> Union[int, None]:
            match = None
            for _ in range(5):
                try:
                    probe = requests.get(self.controller_address + "/list_sessions")
                    if probe.status_code == 200:
                        sessions = probe.json()
                        items = None
                        if isinstance(sessions, dict) and isinstance(sessions.get("sessions"), list):
                            items = sessions.get("sessions")
                        elif isinstance(sessions, list):
                            items = sessions
                        elif isinstance(sessions, dict):
                            try:
                                items = [{"id": int(k), **v} for k, v in sessions.items() if isinstance(v, dict)]
                            except Exception:
                                items = [{"id": k, **v} for k, v in sessions.items() if isinstance(v, dict)]
                        if isinstance(items, list):
                            candidate = _select_sid(items)
                            if candidate is not None:
                                match = candidate
                                break
                except Exception:
                    pass
                time.sleep(0.2)
            return match or current_sid

        sid = _confirm_sid(sid)
        # Brief grace period to avoid controller race on first interact
        time.sleep(0.3)
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
                            messages=output_payload["messages"], tools=(output_payload.get("tools") or [])
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
                # If invalid session id, try one refresh-and-retry cycle
                if "invalid session id" in (result.text or "").lower():
                    new_sid = _confirm_sid(sid)
                    if new_sid is not None and new_sid != sid:
                        sid = new_sid
                        try:
                            result = requests.post(
                                self.controller_address + "/interact",
                                json=InteractRequest(
                                    session_id=sid,
                                    agent_response=response,
                                ).dict(),
                            )
                            if result.status_code == 200:
                                result = result.json()
                                latest_result = result
                                continue
                        except Exception:
                            pass
                requests.post(
                    self.controller_address + "/cancel",
                    json=CancelRequest(session_id=sid).dict(),
                )
                ex = TaskClientException(
                    "interact_failed_http",
                    f"{result.status_code}: {result.text}",
                )
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=str(ex),
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
