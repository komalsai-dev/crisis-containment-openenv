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

# Crisis Containment: Viral Misinformation Graph

A real-world reinforcement learning environment where an AI agent acts as a Crisis Moderator, making strategic decisions to stop the exponential spread of misinformation across a simulated social network graph.

## Motivation

Modern social media platforms face a critical, unsolved challenge: misinformation spreads exponentially. Tools like simple keyword filters are too slow and dumb. Human moderators cannot scale. This environment trains AI agents to act as strategic Crisis Commanders, targeting the most dangerous nodes in a social graph with a limited action budget, while avoiding false positives that penalize innocent accounts.

This is the exact challenge platforms like Meta, X (Twitter), and Reddit deal with in the real world. By training AI agents in this environment, we build systems that can surgically contain misinformation outbreaks without blanket censorship.

## Task Descriptions

### Task 1 - Easy: The Isolated Spam Bot
A single low-follower bot is spamming links with repetitive, clearly harmful content.
- **Objective**: Identify the bot from the feed and apply a Suspend_Account action within the step limit.
- **Expected Score**: 1.00
- **Win Condition**: Bot is suspended without touching innocent accounts.

### Task 2 - Medium: Contextual Intent Recognition
A fake bank-run rumor is spreading. One user is amplifying the panic maliciously. Another user, a journalist, is quoting the same rumor to debunk it.
- **Objective**: Suspend the malicious spreader. Do NOT suspend the journalist.
- **Expected Score**: 1.00
- **Win Condition**: Correct action applied to each user based on intent, not just keywords.

### Task 3 - Hard: Viral Outbreak Containment
A deepfake video is spreading through a large network via a Super Spreader with 10k+ followers. The agent's budget is not enough to suspend every user.
- **Objective**: Prioritize taking down the Super Spreader using Suspend_Account. Use cheaper Add_Context_Warning actions on minor retweeters to drain them.
- **Expected Score**: 0.97+
- **Win Condition**: Maximize network_health score by the end of the episode.

## Observation Space

The agent receives a CrisisContainmentObservation containing:

| Field | Type | Description |
|---|---|---|
| trending_posts | List[PostObs] | List of currently active posts in the feed |
| remaining_budget | int | Remaining action points |
| network_health | float | Global health metric 0.0 to 1.0 |

Each PostObs contains:

| Field | Type | Description |
|---|---|---|
| post_id | str | Unique post identifier |
| user_id | str | Account that made the post |
| text | str | Content of the post |
| follower_count | int | Influence scale of the user |
| virality_velocity | float | Current spreading speed |
| has_context_warning | bool | Whether a warning was applied |

## Action Space

| Action | Budget Cost | Effect |
|---|---|---|
| Ignore | 0 | No change |
| Add_Context_Warning | 1 | Reduces virality by 70%. Best for borderline content. |
| Throttle_User | 2 | Severely limits user reach. Reduces virality by 90%. |
| Suspend_Account | 3 | Removes account. Full +0.5 reward if bot, -0.8 penalty if innocent. |

## Reward Function

Rewards are given at every step (partial progress signals, not just end-of-episode):

- Suspend a bot: +0.50
- Throttle a bot: +0.30
- Add context warning to bot: +0.20
- Suspend an innocent user: -0.80 (large penalty)
- Throttle an innocent user: -0.40
- Exceed budget: -0.10
- Final step bonus: +network_health score (0.0 to 1.0)

## Baseline Scores

Baseline agent using Qwen/Qwen2.5-72B-Instruct via Hugging Face Inference API:

| Task | Score |
|---|---|
| easy | 1.000 |
| medium | 1.000 |
| hard | 0.979 |

## Setup Instructions

```bash
pip install openenv-core openai huggingface_hub
```

### Run the baseline inference script

```bash
python inference.py
```

This runs the environment against all 3 tasks and outputs the required [START], [STEP], and [END] structured logs.

### Environment Variables

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token_here
```

### Run the server locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
crisis_containment/
|-- __init__.py          # Module exports
|-- README.md            # This file
|-- openenv.yaml         # OpenEnv manifest
|-- pyproject.toml       # Project metadata
|-- models.py            # Pydantic Action and Observation models
|-- inference.py         # Official baseline inference script
|-- client.py            # CrisisContainmentEnv client
|-- validate-submission.sh # Pre-submission validator
|-- server/
|   |-- crisis_containment_environment.py  # Core environment and task logic
|   |-- app.py                              # FastAPI server
|   |-- Dockerfile                          # Container definition
```

## API Endpoints

- POST /reset - Reset the environment and get initial observation
- POST /step - Send an action and receive next observation + reward
- GET /state - Get current internal state
- GET /docs - Swagger API documentation
- GET /health - Health check
- WS /ws - WebSocket for low-latency persistent sessions

