# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Crisis Containment Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CrisisContainmentAction, CrisisContainmentObservation


class CrisisContainmentEnv(
    EnvClient[CrisisContainmentAction, CrisisContainmentObservation, State]
):
    """
    Client for the Crisis Containment Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CrisisContainmentEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CrisisContainmentAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CrisisContainmentEnv.from_docker_image("crisis_containment-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CrisisContainmentAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CrisisContainmentAction) -> Dict:
        """
        Convert CrisisContainmentAction to JSON payload for step message.

        Args:
            action: CrisisContainmentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "target_id": action.target_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CrisisContainmentObservation]:
        """
        Parse server response into StepResult[CrisisContainmentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CrisisContainmentObservation
        """
        obs_data = payload.get("observation", {})
        observation = CrisisContainmentObservation(
            trending_posts=obs_data.get("trending_posts", []),
            remaining_budget=obs_data.get("remaining_budget", 0),
            network_health=obs_data.get("network_health", 1.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
