# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Crisis Containment Environment.

The crisis_containment environment is a simple test environment that echoes back messages.
"""

from typing import Literal, List, Dict, Optional, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class PostObs(Observation):
    post_id: str = Field(..., description="Unique ID of the post")
    user_id: str = Field(..., description="ID of the user who made the post")
    text: str = Field(..., description="Content of the post")
    follower_count: int = Field(..., description="Follower count of the user")
    virality_velocity: float = Field(..., description="Current speed at which this post is spreading")
    has_context_warning: bool = Field(default=False, description="Whether this post currently has a context warning")

class CrisisContainmentAction(Action):
    """Action for the Crisis Containment environment."""
    action_type: Literal["Ignore", "Add_Context_Warning", "Throttle_User", "Suspend_Account"] = Field(
        ..., description="The type of action to take."
    )
    target_id: Optional[str] = Field(
        default=None, description="The ID of the user or post to target. For Ignore, this can be None. For Add_Context_Warning, use post_id. For Throttle_User or Suspend_Account, use user_id."
    )

class CrisisContainmentObservation(Observation):
    """Observation from the Crisis Containment environment."""
    trending_posts: List[PostObs] = Field(..., description="A list of currently trending posts in the network")
    remaining_budget: int = Field(..., description="The remaining action budget. Suspend=3, Throttle=2, Context Warning=1, Ignore=0")
    network_health: float = Field(..., description="Global metric from 0.0 to 1.0 indicating network health (1.0 = no misinformation, 0.0 = completely overrun)")
