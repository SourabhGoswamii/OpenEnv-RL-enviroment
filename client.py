# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Taskmanager Environment Client."""

from typing import Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TaskmanagerAction, TaskmanagerObservation


class TaskmanagerEnv(EnvClient[TaskmanagerAction, TaskmanagerObservation, State]):
    """
    Client for the Task Scheduling Environment.

    Supports:
    - default reset()
    - custom reset with user-defined tasks
    """

    # ================= STEP =================

    def _step_payload(self, action: TaskmanagerAction) -> Dict:
        return {
            "task_id": action.task_id,
        }

    # ================= PARSE RESULT =================

    def _parse_result(self, payload: Dict) -> StepResult[TaskmanagerObservation]:
        obs_data = payload.get("observation", {})

        observation = TaskmanagerObservation(
            tasks=obs_data.get("tasks", []),
            current_time=obs_data.get("current_time", 0),
            steps_left=obs_data.get("steps_left", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ================= PARSE STATE =================

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # ================= CUSTOM RESET =================

    async def reset(
        self, tasks: Optional[List[Dict]] = None, **kwargs
    ) -> StepResult[TaskmanagerObservation]:
        """
        Reset the environment. If tasks are provided, they will be used instead of the predefined task list.
        """
        payload = kwargs.pop("config", {}) or {}
        if tasks is not None:
            payload["tasks"] = tasks

        # Call underlying reset via super
        return await super().reset(config=payload, **kwargs)
