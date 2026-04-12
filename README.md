---
title: Crisis Containment Viral Misinformation Graph
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# 🛡️ Crisis Containment: Viral Misinformation Graph

A real-world reinforcement learning environment where an AI agent acts as a **Crisis Moderator**, making strategic decisions to stop the exponential spread of misinformation across a simulated social network graph.

## Motivation

Modern social media platforms face a critical, unsolved challenge: misinformation spreads exponentially. Simple keyword filters are too slow and dumb. Human moderators cannot scale. This environment trains AI agents to act as strategic **Crisis Commanders**, targeting the most dangerous nodes in a social graph with a limited action budget, while avoiding false positives that penalize innocent accounts.

This is the exact challenge platforms like **Meta, X (Twitter), and Reddit** deal with in the real world. By training AI agents in this environment, we build systems that can surgically contain misinformation outbreaks without blanket censorship.

## Task Descriptions

### Task 1 — Easy: The Isolated Spam Bot
A single low-follower bot is spamming links with repetitive, clearly harmful content.
- **Objective**: Identify the bot from the feed and apply a `Suspend_Account` action within the step limit.
- **Expected Score**: 0.99
- **Win Condition**: Bot is suspended without touching innocent accounts.

### Task 2 — Medium: Contextual Intent Recognition
A fake bank-run rumor is spreading. One user is amplifying the panic maliciously. Another user, a journalist, is quoting the same rumor to debunk it.
- **Objective**: Suspend the malicious spreader. Do NOT suspend the journalist.
- **Expected Score**: 0.99
- **Win Condition**: Correct action applied to each user based on intent, not just keywords.

### Task 3 — Hard: Viral Outbreak Containment
A deepfake video is spreading through a large network via a Super Spreader with 10k+ followers. The agent's budget is not enough to suspend every user.
- **Objective**: Prioritize taking down the Super Spreader using `Suspend_Account`. Use cheaper `Add_Context_Warning` actions on minor retweeters.
- **Expected Score**: 0.97+
- **Win Condition**: Maximize `network_health` score by the end of the episode.

## Observation Space

The agent receives a `CrisisContainmentObservation` containing:

| Field | Type | Description |
|---|---|---|
| `trending_posts` | `List[PostObs]` | List of currently active posts in the feed |
| `remaining_budget` | `int` | Remaining action points |
| `network_health` | `float` | Global health metric 0.0 to 1.0 |

Each `PostObs` contains:

| Field | Type | Description |
|---|---|---|
| `post_id` | `str` | Unique post identifier |
| `user_id` | `str` | Account that made the post |
| `text` | `str` | Content of the post |
| `follower_count` | `int` | Influence scale of the user |
| `virality_velocity` | `float` | Current spreading speed |
| `has_context_warning` | `bool` | Whether a warning was applied |

## Action Space

| Action | Budget Cost | Effect |
|---|---|---|
| `Ignore` | 0 | No change |
| `Add_Context_Warning` | 1 | Reduces virality by 70%. Best for borderline content. |
| `Throttle_User` | 2 | Severely limits user reach. Reduces virality by 90%. |
| `Suspend_Account` | 3 | Removes account. +0.5 reward if bot, -0.8 penalty if innocent. |

## Reward Function

Rewards are given at every step (partial progress signals, not just end-of-episode). The internal RL reward signal is designed to guide the agent toward correct decisions:

| Action | Internal Reward Signal |
|---|---|
| Suspend a confirmed bot | +0.50 |
| Throttle a confirmed bot | +0.30 |
| Add context warning to bot | +0.20 |
| Suspend an innocent user | -0.80 (heavy penalty) |
| Throttle an innocent user | -0.40 |
| Exceed budget | -0.10 |
| Final step bonus | +`network_health` score |

> **Note**: All internal reward signals are clamped to the strict open interval **(0.01, 0.99)** before being returned to the agent and evaluator, ensuring full compliance with the OpenEnv scoring specification.

## Baseline Scores

Baseline agent using `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Router:

| Task | Score |
|---|---|
| easy | 0.990 |
| medium | 0.990 |
| hard | 0.979 |

## Setup Instructions

### 1. Install Dependencies

```bash
pip install openenv-core openai pydantic fastapi uvicorn gradio huggingface_hub
```

### 2. Set Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token_here"
export CRISIS_TASK="easy"   # Options: easy | medium | hard
```

### 3. Run the Baseline Inference Script

```bash
python inference.py
```

This runs the environment against all 3 tasks (easy, medium, hard) and outputs the required `[START]`, `[STEP]`, and `[END]` structured logs.

### 4. Run the Server Locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the Server

Once running, verify all endpoints are responding:

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{}"

# Take a step (suspend bot u1)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "Suspend_Account", "target_id": "u1"}'

# Get current state
curl http://localhost:8000/state
```

### 6. Run with Docker

```bash
# Build the Docker image
docker build -t crisis-containment .

# Run the container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  -e CRISIS_TASK=easy \
  crisis-containment
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Reset the environment and get initial observation |
| `POST` | `/step` | Send an action and receive next observation + reward |
| `GET` | `/state` | Get current internal environment state |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Swagger API documentation |
| `WS` | `/ws` | WebSocket for low-latency persistent sessions |

## Project Structure

```
crisis_containment/
├── README.md                    # This file
├── Dockerfile                   # Container definition (root level)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Project metadata and dependencies
├── models.py                    # Pydantic Action and Observation models
├── inference.py                 # Official baseline inference script
├── client.py                    # CrisisContainmentEnv client
├── __init__.py                  # Module exports
└── server/
    ├── app.py                   # FastAPI server with custom Gradio UI
    ├── crisis_containment_environment.py  # Core environment and task logic
    └── requirements.txt         # Server dependencies
```

## Connect to This Environment

From Python using the OpenEnv client:

```python
from client import CrisisContainmentEnv

env = CrisisContainmentEnv(base_url="https://komalsai777-crisis-containment.hf.space")
obs = env.reset()
print(obs)
```

Or connect to a locally running server:

```python
env = CrisisContainmentEnv(base_url="http://localhost:8000")
```
