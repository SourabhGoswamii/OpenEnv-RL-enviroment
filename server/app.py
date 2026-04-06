# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Taskmanager Environment.
"""

from fastapi.responses import HTMLResponse
from fastapi import APIRouter
import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import TaskmanagerAction, TaskmanagerObservation
    from .taskmanager_environment import TaskmanagerEnvironment
except ImportError:
    from models import TaskmanagerAction, TaskmanagerObservation
    from server.taskmanager_environment import TaskmanagerEnvironment


# ================= CREATE APP =================

app = create_app(
    TaskmanagerEnvironment,
    TaskmanagerAction,
    TaskmanagerObservation,
    env_name="taskmanager",
    max_concurrent_envs=1,
)

# ================= ROUTER =================

router = APIRouter()

# 🔥 Serve demo UI
@router.get("/", response_class=HTMLResponse)
def home():
    file_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 🔥 Run full agent (for UI)
@router.get("/run-agent")
def run_agent():
    env = TaskmanagerEnvironment()

    obs = env.reset()
    total_reward = 0
    steps = []

    for _ in range(20):
        tickets = obs.tasks

        if not tickets:
            break

        # 🔥 same logic as inference
        def score(t):
            type_score = {"bug": 3, "feature": 2, "enhancement": 1}
            return (
                type_score.get(t["type"], 0),
                t["priority"],
                -t["deadline"]
            )

        best = sorted(tickets, key=score, reverse=True)[0]

        obs = env.step(type("obj", (), {"task_id": best["id"]}))

        total_reward += obs.reward

        steps.append({
            "chosen": best,
            "remaining": obs.tasks,
            "reward": obs.reward,
            "time": obs.current_time
        })

        if obs.done:
            break

    # 🔥 normalize score
    score = max(0.0, min(1.0, total_reward / 200))

    return {
        "steps": steps,
        "final_score": score
    }


app.include_router(router)

# ================= MAIN =================

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()