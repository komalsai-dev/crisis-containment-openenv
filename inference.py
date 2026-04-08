"""
Inference Script Example for Crisis Containment
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from models import CrisisContainmentAction, CrisisContainmentObservation
from server.crisis_containment_environment import CrisisContainmentEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "crisis_containment"
MAX_STEPS = 10
TEMPERATURE = 0.5
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Crisis Commander Agent. Your job is to stop misinformation spreading in a social network.
    You will be shown trending posts. 
    You have a budget for actions:
    Suspend_Account = 3
    Throttle_User = 2
    Add_Context_Warning = 1
    Ignore = 0
    
    You must output a JSON string with two keys: "action_type" and "target_id".
    Example: {"action_type": "Suspend_Account", "target_id": "u1"}
    Only output the JSON object, nothing else.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: CrisisContainmentObservation, last_reward: float) -> str:
    feed = json.dumps([p.model_dump() for p in obs.trending_posts], indent=2)
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Remaining Budget: {obs.remaining_budget}
        Current Network Health: {obs.network_health:.2f}
        Trending Posts:
        {feed}
        
        Decide your next action.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, obs: CrisisContainmentObservation, last_reward: float) -> dict:
    user_prompt = build_user_prompt(step, obs, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
        parsed = json.loads(text)
        return parsed
    except Exception as exc:
        print(f"[DEBUG] Model request failed or parse error: {exc}", flush=True)
        return {"action_type": "Ignore", "target_id": None}

async def main(task_name: str) -> None:
    # Handle mock scenario (no api key)
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    else:
        client = None

    env = CrisisContainmentEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(obs, "done", False):
                break

            if client:
                action_data = get_model_message(client, step, obs, last_reward)
            else:
                # Mock dummy logic
                if task_name == "easy":
                    action_data = {"action_type": "Suspend_Account", "target_id": "u1"}
                elif task_name == "medium":
                    action_data = {"action_type": "Suspend_Account", "target_id": "u1"}
                else:
                    action_data = {"action_type": "Add_Context_Warning", "target_id": "p1"}
            
            action = CrisisContainmentAction(action_type=action_data.get("action_type", "Ignore"), target_id=action_data.get("target_id"))
            action_str = f"{action.action_type}({action.target_id})"

            obs = env.step(action)
            reward = obs.reward or 0.0
            done = getattr(obs, "done", False)
            error = None
            if hasattr(obs, "metadata") and obs.metadata and "error" in obs.metadata:
                error = obs.metadata["error"]

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate final score based on health and accumulated rewards?
        # Actually in env, the final reward incorporates health. Normalizing for the hackathon [0, 1]:
        # Using the last observation's network_health as the primary metric, bounded strictly to (0,1)
        score = getattr(obs, "network_health", sum(rewards))
        score = min(max(score, 0.001), 0.999)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        os.environ["CRISIS_TASK"] = task
        asyncio.run(main(task))
