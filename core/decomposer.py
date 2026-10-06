import json
import re
from typing import Dict, Any
from core.ollama import OllamaManager


class TaskDecomposer:
    def __init__(self, ollama: OllamaManager):
        self.ollama = ollama

    def decompose_goal(self, user_instruction: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a task decomposer for a system administration agent.\n"
            "Analyze the user request and break it down into a sequential list of high-level atomic goals "
            "ONLY IF it requires multiple separate operations.\n\n"
            "CRITICAL RULES:\n"
            "1. If the request is a simple, single direct action (e.g., 'list files in folder', 'check uptime', 'show memory usage'), "
            "DO NOT split it. Return exactly ONE step containing the raw user request.\n"
            "2. Do NOT add meta-instructions like 'open a terminal', 'type command', or 'review output'.\n"
            "3. Output ONLY a valid JSON object matching this schema:\n"
            '{\n  "goal": "summary of user goal",\n  "steps": ["step description"]\n}'
        )

        # End prompt cleanly without placing a stop-token keyword at the end
        prompt = f"User Task: {user_instruction}\n\nJSON Output:"

        try:
            raw_resp = self.ollama.generate(
                role="decomposer",
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                    "stop": ["User:", "\n\nUser"]
                }
            )

            if not raw_resp or not raw_resp.strip():
                raise ValueError("Model returned an empty response string.")

            # Strip code fencing if present
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json.strip(), flags=re.MULTILINE)
            clean_json = clean_json.strip()

            plan = json.loads(clean_json)
            
            if not isinstance(plan.get("steps"), list) or len(plan["steps"]) == 0:
                plan["steps"] = [user_instruction]

            return plan

        except Exception as e:
            print(f"[Decomposer] JSON parse or execution failed ({e}), falling back to direct instruction.")
            return {
                "goal": user_instruction,
                "steps": [user_instruction]
            }