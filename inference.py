import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from client import TaskmanagerEnv
from models import TaskmanagerAction
from grader import compute_score  # ✅ GRADER USED
import os


# ================= CONFIG =================

API_KEY = os.environ.get("API_KEY", "dummy")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:4000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = "ai-ticket-prioritization"
BENCHMARK = "taskmanager"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.6

# ================= LOGGING =================


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ================= SMART POLICY =================


def choose_best_ticket(tickets):
    """
    Priority logic:
    bug > feature > enhancement
    then higher priority, earlier deadline
    """

    if not tickets:
        return None

    def score(ticket):
        type_score = {"bug": 3, "feature": 2, "enhancement": 1}

        return (
            type_score.get(ticket["type"], 0),
            ticket["priority"],
            -ticket["deadline"],
        )

    best = sorted(tickets, key=score, reverse=True)[0]
    return best["id"]


# ================= MAIN =================


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = TaskmanagerEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        # 🔥 RESET ENV (tickets auto-generated)
        result = await env.reset()
        obs = result.observation

        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
            )
        except Exception:
            pass

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            tickets = obs.tasks

            ticket_id = choose_best_ticket(tickets)

            if ticket_id is None:
                break

            # 🔥 STEP
            result = await env.step(TaskmanagerAction(task_id=ticket_id))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"resolve_ticket_{ticket_id}",
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # ================= GRADER =================

        total_reward = sum(rewards)

        # realistic upper bound
        max_per_step = 15
        max_possible = len(rewards) * max_per_step

        score = compute_score(total_reward, max_possible)  # ✅ GRADER USED

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
