# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Crisis Containment Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import os
import math
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CrisisContainmentAction, CrisisContainmentObservation, PostObs
except ImportError:
    from models import CrisisContainmentAction, CrisisContainmentObservation, PostObs

class CustomState(State):
    internal_nodes: dict = {}
    current_task: str = "easy"
    step_limit: int = 10
    budget: int = 10
    
class CrisisContainmentEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = CustomState(episode_id=str(uuid4()), step_count=0)

    def _setup_task(self, task_name: str):
        nodes = {}
        if task_name == "easy":
            # 1 spam bot, 1 innocent
            nodes = {
                "u1": {"is_bot": True, "text": "Click here for free movie!! [Link]", "followers": 10, "virality": 2.0, "status": "active", "has_warning": False, "post_id": "p1"},
                "u2": {"is_bot": False, "text": "Lovely day out today.", "followers": 50, "virality": 0.1, "status": "active", "has_warning": False, "post_id": "p2"},
            }
            budget = 5
            steps = 3
        elif task_name == "medium":
            # 1 malicious spreader, 1 journalist, 1 innocent
            nodes = {
                "u1": {"is_bot": True, "text": "BREAKING: Hackers just stole your data from the bank! Withdraw now!", "followers": 500, "virality": 5.0, "status": "active", "has_warning": False, "post_id": "p1"},
                "u2": {"is_bot": False, "text": "The rumors about the bank hack are completely fake. I confirmed with their CTO.", "followers": 2000, "virality": 3.0, "status": "active", "has_warning": False, "post_id": "p2"},
                "u3": {"is_bot": False, "text": "Can someone recommend a good coffee shop?", "followers": 100, "virality": 0.2, "status": "active", "has_warning": False, "post_id": "p3"},
            }
            budget = 5
            steps = 5
        else: # hard
            # small outbreak network
            nodes = {
                "super_spreader": {"is_bot": True, "text": "[DEEPFAKE VIDEO] The market is crashing!!", "followers": 10000, "virality": 10.0, "status": "active", "has_warning": False, "post_id": "p1"},
                "retweeter_1": {"is_bot": False, "text": "OMG just saw the video, is this real?", "followers": 500, "virality": 2.0, "status": "active", "has_warning": False, "post_id": "p2"},
                "retweeter_2": {"is_bot": False, "text": "Selling all my stocks right now.", "followers": 200, "virality": 1.5, "status": "active", "has_warning": False, "post_id": "p3"},
                "innocent_1": {"is_bot": False, "text": "Just ate a sandwich.", "followers": 50, "virality": 0.1, "status": "active", "has_warning": False, "post_id": "p4"},
            }
            budget = 4 # Not enough to suspend everyone (suspend=3). Must suspend super spreader, maybe context warning retweeters.
            steps = 6

        self._state.internal_nodes = nodes
        self._state.budget = budget
        self._state.step_limit = steps
        self._state.current_task = task_name

    def _get_observation(self) -> CrisisContainmentObservation:
        posts = []
        health = 1.0
        bad_virality = 0.0
        total_virality = 0.0

        for uid, data in self._state.internal_nodes.items():
            if data["status"] in ["suspended", "throttled"]:
                continue
            
            total_virality += data["virality"]
            if data["is_bot"]:
                bad_virality += data["virality"]

            posts.append(
                PostObs(
                    post_id=data["post_id"],
                    user_id=uid,
                    text=data["text"],
                    follower_count=data["followers"],
                    virality_velocity=data["virality"],
                    has_context_warning=data["has_warning"]
                )
            )

        if total_virality > 0:
            health = max(0.0, 1.0 - (bad_virality / (total_virality + 0.1)))

        return CrisisContainmentObservation(
            trending_posts=posts,
            remaining_budget=self._state.budget,
            network_health=health,
            done=False,
            reward=0.0
        )

    def reset(self) -> CrisisContainmentObservation:
        self._state = CustomState(episode_id=str(uuid4()), step_count=0)
        # Read task from ENV or default to medium
        task_level = os.getenv("CRISIS_TASK", "easy").lower()
        self._setup_task(task_level)
        return self._get_observation()

    def step(self, action: CrisisContainmentAction) -> CrisisContainmentObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        done = False
        error = ""

        cost_map = {
            "Suspend_Account": 3,
            "Throttle_User": 2,
            "Add_Context_Warning": 1,
            "Ignore": 0
        }
        action_cost = cost_map.get(action.action_type, 0)

        if action_cost > self._state.budget:
            # Invalid action due to budget
            error = f"Insufficient budget for {action.action_type}"
            obs = self._get_observation()
            obs.reward = -0.1
            obs.metadata = {"error": error}
            if self._state.step_count >= self._state.step_limit:
                obs.done = True
            return obs

        self._state.budget -= action_cost

        # Apply action
        if action.action_type != "Ignore":
            uid_to_target = action.target_id
            # If target_id is a post_id, find the user
            if action.action_type == "Add_Context_Warning":
                for u, d in self._state.internal_nodes.items():
                    if d["post_id"] == action.target_id:
                        uid_to_target = u
                        break

            if uid_to_target in self._state.internal_nodes:
                node = self._state.internal_nodes[uid_to_target]
                
                if action.action_type == "Suspend_Account":
                    node["status"] = "suspended"
                    if node["is_bot"]:
                        reward += 0.5 # Good! Suspended bad guy
                    else:
                        reward -= 0.8 # Terrible! Suspended innocent

                elif action.action_type == "Throttle_User":
                    node["status"] = "throttled"
                    node["virality"] *= 0.1
                    if node["is_bot"]:
                        reward += 0.3
                    else:
                        reward -= 0.4

                elif action.action_type == "Add_Context_Warning":
                    node["has_warning"] = True
                    node["virality"] *= 0.3
                    if node["is_bot"]:
                        reward += 0.2
                    elif "fake" in node["text"].lower(): # Journalist debunking
                        reward += 0.1 # Good to add context but not strictly harmful
                    else:
                        reward -= 0.1

        # Simulate spread: unmitigated bots increase in virality
        for u, d in self._state.internal_nodes.items():
            if d["status"] == "active" and d["is_bot"] and not d["has_warning"]:
                d["virality"] *= 1.5

        if self._state.step_count >= self._state.step_limit:
            done = True
            
            # Final scoring (Grader logic embedded)
            health = self._get_observation().network_health
            # Final normalized score from 0.0 to 1.0 based on health
            reward += health

        obs = self._get_observation()
        # Strictly bound the reward returned via HTTP/internal to (0, 1) per judging rules
        obs.reward = min(max(reward, 0.001), 0.999)
        obs.done = done
        return obs

    @property
    def state(self) -> CustomState:
        return self._state
