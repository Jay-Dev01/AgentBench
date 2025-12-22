import asyncio
import logging
import re
from typing import Optional, Tuple
from agentrl.worker.environment import EnvironmentController


class SWEBenchContainer:
    """
    Wrapper for SWE-bench Docker container operations.
    Similar to os_interaction Container but adapted for SWE-bench.
    """

    def __init__(self, controller: EnvironmentController, subtype: str, image: str):
        self.subtype = subtype
        self.image = image
        self.controller = controller
        self.session_id: Optional[str] = None
        self.container_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize container session."""
        self.logger.info(f"Initializing container with image: {self.image}")
        # Tell the environment delegation which image to use
        self.controller.delegation.current_image = self.image
        res = await self.controller.start_session(self.subtype)
        self.session_id = res[0]
        self.container_id = res[1][self.subtype]
        self.logger.info(f"Container initialized: {self.container_id}")

    async def cleanup(self):
        """Cleanup container session."""
        if self.session_id:
            self.logger.info(f"Cleaning up container: {self.container_id}")
            await self.controller.end_session(self.session_id)

    async def execute_bash(self, command: str, timeout: int = 60) -> str:
        """
        Execute bash command in container.

        Args:
            command: Bash command to execute
            timeout: Timeout in seconds

        Returns:
            Command output (stdout + stderr)
        """
        await self.controller.renew_session(self.session_id)

        try:
            output = await asyncio.wait_for(
                self.controller.execute_shell(self.container_id, command),
                timeout=timeout
            )

            # Clean terminal control sequences (same as os_interaction)
            output = re.sub(b"\x1b.+@.+[#|$] ", b'', output)
            output = re.sub(b'\x1b\\[[0-9;]*[a-zA-Z]', b'', output)
            output = re.sub(b'\x1b][0-9]*;[^\x07]*\x07', b'', output)
            output = re.sub(b'\x1b\[\?2004[hl]', b'', output)
            output = re.sub(b'\x07', b'', output)

            return output.decode('utf-8', errors='replace')

        except asyncio.TimeoutError:
            self.logger.warning(f"Command timed out after {timeout}s: {command[:100]}")
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"Error executing command: {e}", exc_info=True)
            return f"Error: {str(e)}"

    async def read_file(self, file_path: str) -> str:
        """Read file contents."""
        command = f"cat {file_path}"
        return await self.execute_bash(command)

    async def search_code(self, pattern: str, directory: str = '.') -> str:
        """Search for pattern in code."""
        # Use grep with context
        command = f"cd /testbed && grep -rn --include='*.py' '{pattern}' {directory} 2>/dev/null || echo 'No matches found'"
        return await self.execute_bash(command)
