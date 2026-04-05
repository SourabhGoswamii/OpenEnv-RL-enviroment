# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
AI Ticket Prioritization Environment (Jira-like)

Simulates a real-world engineering workflow:
- Bug fixes (critical)
- Feature development
- UI enhancements

Agent must prioritize tickets to maximize business impact and avoid SLA violations.
"""

from dataclasses import dataclass
from uuid import uuid4
import random

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:

    class Environment:
        pass

    @dataclass
    class State:
        episode_id: str
        step_count: int


try:
    from ..models import TaskmanagerAction, TaskmanagerObservation
except ImportError:
    from models import TaskmanagerAction, TaskmanagerObservation


class TaskmanagerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_time = 0
        self.tickets = []
        self.max_steps = 20

        self.total_reward = 0
        self.episode_count = 0

    # ================= TICKET GENERATOR =================

    def generate_tickets(self, num_tickets):
        tickets = []
        current_time = 0

        for i in range(num_tickets):
            ticket_type = random.choice(["bug", "feature", "enhancement"])

            effort = random.randint(1, 3)

            # 🔥 ensure feasible deadline
            slack = random.randint(3, 8)
            deadline = current_time + effort + slack

            priority = random.randint(1, 5)

            ticket = {
                "id": i + 1,
                "deadline": deadline,
                "priority": priority,
                "effort": effort,
                "type": ticket_type,
            }

            tickets.append(ticket)

            # update time so sequence is solvable
            current_time += effort

        # 🔥 shuffle so agent must think
        random.shuffle(tickets)

        return tickets

    # ================= RESET =================

    def reset(self) -> TaskmanagerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_time = 0

        self.episode_count += 1

        avg_reward = self.total_reward / max(1, self.episode_count)

        if avg_reward < 5:
            num_tickets = 5
        elif avg_reward < 15:
            num_tickets = 8
        else:
            num_tickets = 12

        self.tickets = self.generate_tickets(num_tickets)

        print(
            f"Episode {self.episode_count} | Tickets: {num_tickets} | Avg reward: {avg_reward:.2f}"
        )

        return TaskmanagerObservation(
            tasks=self.tickets,  # ⚠️ keep 'tasks' for compatibility
            current_time=self.current_time,
            steps_left=self.max_steps,
            reward=0.0,
            done=False,
        )

    # ================= STEP =================

    def step(self, action: TaskmanagerAction) -> TaskmanagerObservation:
        self._state.step_count += 1

        reward = 0

        ticket = next((t for t in self.tickets if t["id"] == action.task_id), None)

        if ticket:
            self.current_time += ticket["effort"]

            # 🎯 BASE REWARD
            if self.current_time <= ticket["deadline"]:
                reward = ticket["priority"] * 3  # boosted reward for being on time
            else:
                delay = self.current_time - ticket["deadline"]
                # Soft penalty: base priority minus a small delay fraction (can still be positive if slightly late)
                reward = max(-2, ticket["priority"] - (delay * 0.5))

            # 🔥 TYPE MULTIPLIER (REAL-WORLD LOGIC)
            if ticket["type"] == "bug":
                reward *= 2  # critical
            elif ticket["type"] == "feature":
                reward *= 1.5
            else:  # enhancement
                reward *= 1

            # remove ticket
            self.tickets = [t for t in self.tickets if t["id"] != action.task_id]

        else:
            reward = -1

        self.total_reward += reward

        done = len(self.tickets) == 0 or self._state.step_count >= self.max_steps

        return TaskmanagerObservation(
            tasks=self.tickets,
            current_time=self.current_time,
            steps_left=self.max_steps - self._state.step_count,
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "remaining_tickets": len(self.tickets),
            },
        )

    @property
    def state(self) -> State:
        return self._state
