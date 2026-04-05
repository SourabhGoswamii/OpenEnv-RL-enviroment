# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Taskmanager Environment.

The taskmanager environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Dict


class TaskmanagerAction(Action):
    """Action for the Taskmanager environment - just a message to echo."""

    task_id: int = Field(..., description="Task to execute")


class TaskmanagerObservation(Observation):
    """Observation for Task Scheduling Environment"""

    tasks: List[Dict] = Field(
        default_factory=list, description="List of remaining tickets"
    )
    current_time: int = Field(default=0, description="Current time in the schedule")
    steps_left: int = Field(default=0, description="Steps remaining in episode")
