from typing import List

from agentrl.worker.environment import EnvironmentDelegation


class SWEBenchRebenchEnvironmentDelegation(EnvironmentDelegation):
    """
    Environment delegation for SWE-rebench task.
    Manages Docker containers with pre-built SWE-bench images.
    """

    def __init__(self):
        super().__init__('swebench_rebench')
        self.image_cache = {}  # Cache of pulled images
        self.current_image = None  # Will be set by container before start_session

    def get_subtypes(self) -> List[str]:
        # Dynamic subtypes based on docker images from dataset
        # We use 'swerebench' as the main subtype
        return ['swerebench']

    def is_exclusive(self, subtype: str) -> bool:
        # Each container is exclusive (one per instance)
        return True

    async def create_docker_container(self, attrs: dict, subtype: str) -> dict:
        """
        Configure Docker container attributes for SWE-bench instance.

        The image will be specified dynamically when starting the container
        based on the instance's docker_image field.
        """
        # Set the Docker image from current_image (set by container before start_session)
        if self.current_image:
            attrs['Image'] = self.current_image
        else:
            raise ValueError("No Docker image specified. current_image must be set before creating container.")

        attrs['User'] = 'root'
        attrs['WorkingDir'] = '/testbed'
        attrs['AttachStdin'] = False
        attrs['AttachStdout'] = False
        attrs['AttachStderr'] = False

        # Resource limits
        attrs['HostConfig']['Memory'] = 4 * 1024 * 1024 * 1024  # 4GB RAM
        attrs['HostConfig']['MemorySwap'] = attrs['HostConfig']['Memory']
        attrs['HostConfig']['NanoCpus'] = 2_000_000_000  # 2 vCPUs

        # Mount tmpfs for faster I/O
        attrs['HostConfig']['Tmpfs'] = {
            '/tmp': 'rw,exec,nosuid,size=1g'
        }

        return attrs

    def get_concurrency_limit(self, subtype: str) -> int:
        # Limit concurrent containers (resource intensive)
        return 8

    def get_reuse_limit(self, subtype: str) -> int:
        # Don't reuse containers (need clean state per instance)
        return 1
