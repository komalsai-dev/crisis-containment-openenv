# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Crisis Containment Environment.

This module creates an HTTP server that exposes the CrisisContainmentEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from datetime import datetime
import gradio as gr

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CrisisContainmentAction, CrisisContainmentObservation
    from .crisis_containment_environment import CrisisContainmentEnvironment
except ImportError:
    from models import CrisisContainmentAction, CrisisContainmentObservation
    from server.crisis_containment_environment import CrisisContainmentEnvironment


def crisis_gradio_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Custom premium Gradio Builder for Crisis Containment."""
    with gr.Blocks(title="Crisis Containment Agent", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown(f"# 🛡️ Crisis Containment: Viral Misinformation Graph")
        gr.Markdown("Welcome to the Crisis Containment Operations Center. Use the Human/Agent testing interface below to step through an outbreak simulation.")
        
        with gr.Row():
            # LEFT: Actions
            with gr.Column(scale=1):
                gr.Markdown("### 🕹️ Action Command Terminal")
                step_inputs = []
                for field in action_fields:
                    name = field["name"]
                    label = name.replace("_", " ").title()
                    choices = field.get("choices")
                    if choices:
                        inp = gr.Dropdown(choices=choices, label=label, value=choices[0])
                    else:
                        inp = gr.Textbox(label=label, placeholder=f"Enter {label} (e.g., u1, p2)")
                    step_inputs.append(inp)
                
                step_btn = gr.Button("Execute Action", variant="primary")
                
                with gr.Row():
                    reset_btn = gr.Button("Deploy New Outbreak (Reset)", variant="secondary")
                    get_state_btn = gr.Button("Fetch Raw Internal State", variant="secondary")
                
                gr.Markdown("### 📡 Global Status")
                status_text = gr.Markdown("Status: *Awaiting deployment...*")
                state_display = gr.JSON(label="State Dump")

            # RIGHT: Observations & Threat Level
            with gr.Column(scale=2):
                gr.Markdown("### 🌐 Live Network Observer")
                with gr.Accordion("Current Observation Feed", open=True):
                    obs_display = gr.JSON(label="Network Observation Feed")
                    reward_display = gr.Number(label="Latest Agent Reward", interactive=False)
                
                with gr.Accordion("Incident Action Log", open=True):
                    history_log = gr.HTML(value="<div style='color:gray'>No containment actions logged yet...</div>")
                
                with gr.Accordion("📚 Complete Documentation (README)", open=False):
                    try:
                        with open("README.md", "r", encoding="utf-8") as f:
                            gr.Markdown(f.read())
                    except Exception:
                        gr.Markdown("*Documentation temporarily unavailable.*")
        
        history_state = gr.State([])

        def update_history(history, action, obs, reward, done):
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = f'''
            <div style="border-left: 4px solid #F44336; padding: 10px; margin: 10px 0; background: #1e1e1e; color: #fff; border-radius: 4px;">
                <small style="color:#bbb">{timestamp} (Step {len(history)+1})</small><br>
                <b>Command Executed:</b> {action}<br>
                <b>Network Health:</b> {obs.get("network_health", "N/A")}<br>
                <b>Remaining Budget:</b> {obs.get("remaining_budget", "N/A")}<br>
                <b style="color:{'#4CAF50' if reward > 0 else '#FF5252'}">Reward: {reward:.3f}</b> {'<span style="color:#FFC107; font-weight:bold;">[CRISIS CONTAINED (DONE)]</span>' if done else ''}
            </div>
            '''
            history.insert(0, entry)
            return history, "".join(history)

        async def on_reset():
            data = await web_manager.reset_environment()
            return (
                data["observation"], 
                0.0, 
                "Status: **Simulation Running - Crisis Detected** 🚨", 
                data,
                [], 
                "<div style='color:gray'>Crisis Environment Reset. Waiting for first containment action...</div>"
            )

        async def on_step(*args):
            action_data = {}
            for i, field in enumerate(action_fields):
                action_data[field["name"]] = args[i]
            
            try:
                data = await web_manager.step_environment(action_data)
                curr_history = args[-1]
                new_hist, hist_html = update_history(
                    curr_history, 
                    action_data, 
                    data["observation"], 
                    data.get("reward", 0), 
                    data.get("done", False)
                )
                
                return (
                    data["observation"], 
                    data.get("reward", 0), 
                    f"Status: {'**Simulation Finished** ✅' if data.get('done') else '**Simulation Running** 🚨'}",
                    data,
                    new_hist,
                    hist_html
                )
            except Exception as e:
                return {}, 0, f"Error: {str(e)}", {}, args[-1], "".join(args[-1])

        def on_get_state():
            return web_manager.get_state()

        reset_btn.click(fn=on_reset, outputs=[obs_display, reward_display, status_text, state_display, history_state, history_log])
        step_btn.click(fn=on_step, inputs=step_inputs + [history_state], outputs=[obs_display, reward_display, status_text, state_display, history_state, history_log])
        get_state_btn.click(fn=on_get_state, outputs=[state_display])
        
    return demo

os.environ["ENABLE_WEB_INTERFACE"] = "true"

# Create the app with web interface and custom builder
app = create_app(
    CrisisContainmentEnvironment,
    CrisisContainmentAction,
    CrisisContainmentObservation,
    env_name="crisis_containment",
    max_concurrent_envs=1,
    gradio_builder=crisis_gradio_builder
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m crisis_containment.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn crisis_containment.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()

