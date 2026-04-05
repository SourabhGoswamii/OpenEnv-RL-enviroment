---
title: Taskmanager Environment Server
emoji: 🎬
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - scheduling
---

# Taskmanager Environment

A reinforcement learning environment that simulates a real-world engineering workflow. The agent must prioritize tickets (bugs, features, UI enhancements) to maximize business impact and avoid SLA violations. 

## Quick Start

The simplest way to use the Taskmanager environment is through the `TaskmanagerEnv` class:

```python
from taskmanager import TaskmanagerAction, TaskmanagerEnv

try:
    # Create environment from Docker image
    env = TaskmanagerEnv.from_docker_image("taskmanager:latest")

    # Reset to start a new episode
    result = env.reset()
    
    print(f"Current Time: {result.observation.current_time}")
    print(f"Available Tasks: {len(result.observation.tasks)}")

    # Execute tasks until the episode is done
    done = False
    while not done:
        # Simple policy: pick the first available task
        if not result.observation.tasks:
            break
            
        task_to_execute = result.observation.tasks[0]
        task_id = task_to_execute["id"]
        
        # Take a step
        result = env.step(TaskmanagerAction(task_id=task_id))
        
        print(f"Executed Task ID: {task_id}")
        print(f"  → Reward: {result.reward}")
        print(f"  → Current Time: {result.observation.current_time}")
        print(f"  → Tasks Remaining: {len(result.observation.tasks)}")
        
        done = result.done

    print("Episode completed!")

finally:
    # Always clean up
    env.close()
```

## Environment Details

### Action
**TaskmanagerAction**: Contains a single field specifying the task to execute.
- `task_id` (int) - The ID of the task/ticket to execute

### Observation
**TaskmanagerObservation**: Contains the current state of the environment.
- `tasks` (List[Dict]) - List of remaining tickets. Each ticket has:
  - `id` (int): Unique identifier
  - `type` (str): "bug", "feature", or "enhancement"
  - `effort` (int): Time required to complete the ticket
  - `priority` (int): Importance (1-5)
  - `deadline` (int): Target completion time
- `current_time` (int) - Current time in the schedule
- `steps_left` (int) - Steps remaining in the episode
- `reward` (float) - Reward received from the previous action
- `done` (bool) - Whether the episode is complete (all tasks done or max steps reached)
- `metadata` (dict) - Additional info like step count

### Reward Function
The reward function is designed to simulate business impact:
1. **Base Reward**: `priority * 3` if completed before the deadline.
2. **Penalty**: If delayed, the reward is reduced based on the delay (`priority - delay * 0.5`), with a minimum of -2.
3. **Type Multipliers**:
   - **Bugs**: 2.0x multiplier (Critical)
   - **Features**: 1.5x multiplier
   - **Enhancements**: 1.0x multiplier
4. **Invalid Action**: -1 reward for attempting a non-existent task ID.

## Building the Docker Image

Before using the environment, you need to build the Docker image.

**To create the Docker image:**
```bash
# From project root
docker build -t taskmanager .
```

**To run the Docker image locally:**
```bash
# Run the container in detached mode and map port 8000
docker run -d -p 8000:8000 --name taskmanager_server taskmanager
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Project Structure

```
taskmanager/
├── client.py              # Environment client implementation
├── models.py              # Action and Observation Pydantic models
├── openenv.yaml           # OpenEnv manifest
├── server/
│   ├── app.py             # FastAPI application
│   └── taskmanager_environment.py  # Core environment logic and reward function
└── Dockerfile             # Container definition
```