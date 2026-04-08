---
title: Crisis Containment Viral Misinformation Graph
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
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


# Crisis Containment Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Crisis Containment environment is through the `CrisisContainmentEnv` class:

```python
from crisis_containment import CrisisContainmentAction, CrisisContainmentEnv

try:
    # Create environment from Docker image
    crisis_containmentenv = CrisisContainmentEnv.from_docker_image("crisis_containment-env:latest")

    # Reset
    result = crisis_containmentenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = crisis_containmentenv.step(CrisisContainmentAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    crisis_containmentenv.close()
```

That's it! The `CrisisContainmentEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t crisis_containment-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**CrisisContainmentAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**CrisisContainmentObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Crisis Containment environment server running, you can connect directly:

```python
from crisis_containment import CrisisContainmentEnv

# Connect to existing server
crisis_containmentenv = CrisisContainmentEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = crisis_containmentenv.reset()
result = crisis_containmentenv.step(CrisisContainmentAction(message="Hello!"))
```

Note: When connecting to an existing server, `crisis_containmentenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from crisis_containment import CrisisContainmentAction, CrisisContainmentEnv

# Connect with context manager (auto-connects and closes)
with CrisisContainmentEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(CrisisContainmentAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    CrisisContainmentEnvironment,  # Pass class, not instance
    CrisisContainmentAction,
    CrisisContainmentObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from crisis_containment import CrisisContainmentAction, CrisisContainmentEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with CrisisContainmentEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(CrisisContainmentAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/crisis_containment_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
crisis_containment/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # CrisisContainmentEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── crisis_containment_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
