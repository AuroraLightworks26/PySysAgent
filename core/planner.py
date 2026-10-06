import json
import re
from typing import Dict, Any, List, Optional
from core.ollama import OllamaManager


class ExecutionPlanner:
    def __init__(self, ollama: OllamaManager):
        self.ollama = ollama

    def generate_plan(
        self,
        goal: str,
        profile_facts: Optional[List[Dict[str, str]]] = None,
        distilled_rules: Optional[List[str]] = None,
        historical_memories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        profile_str = "\n".join([f"- {p.get('category')}.{p.get('attribute')}: {p.get('value')}" for p in (profile_facts or [])]) or "None"
        rules_str = "\n".join([f"- {r}" for r in (distilled_rules or [])]) or "None"
        memories_str = "\n".join([f"- {m}" for m in (historical_memories or [])]) or "None"

        system_prompt = (
            "You are an expert Linux System Administration Execution Planner.\n"
            "Your job is to generate a precise, safe, step-by-step execution plan for a system administration goal.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. Differentiate between 'probe' steps (read-only commands like `status`, `ls`, `grep`, `cat`, `--help`) "
            "   and 'action' steps (state-modifying commands or scripts).\n"
            "2. Always start with a 'probe' phase if the current state of a resource is unknown.\n"
            "3. Enforce strict step dependencies using step IDs.\n"
            "4. Output MUST be strictly valid JSON matching the following schema without conversational text:\n"
            "{\n"
            '  "reasoning": "Short explanation of strategy",\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "1",\n'
            '      "type": "probe",\n'
            '      "command": "ufw status",\n'
            '      "purpose": "Verify current firewall state",\n'
            '      "depends_on": []\n'
            "    }\n"
            "  ],\n"
            '  "mental_model_updates": {\n'
            '    "profile_updates": [{"category": "network", "attribute": "firewall", "value": "ufw"}],\n'
            '    "new_rules": [{"context_key": "ufw_auth", "description": "Requires sudo for status checks"}]\n'
            "  }\n"
            "}\n"
        )

        user_prompt = (
            f"=== SYSTEM CONTEXT ===\n"
            f"System Profile:\n{profile_str}\n\n"
            f"Distilled Directives:\n{rules_str}\n\n"
            f"Past Execution History:\n{memories_str}\n\n"
            f"=== USER GOAL ===\n"
            f"{goal}\n\n"
            f"[END]"
        )

        try:
            raw_resp = self.ollama.generate(
                role="planner",
                prompt=user_prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                    "stop": ["[END]", "User:"]
                }
            )

            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json.strip(), flags=re.MULTILINE)

            plan = json.loads(clean_json)

            if "steps" not in plan or not isinstance(plan["steps"], list):
                plan["steps"] = [{
                    "id": "1",
                    "type": "action",
                    "command": goal,
                    "purpose": "Execute direct request",
                    "depends_on": []
                }]

            return plan

        except Exception as e:
            print(f"[Planner] Error generating plan via LLM ({e}). Falling back to single-step execution.")
            return {
                "reasoning": "Fallback due to planning parse error.",
                "steps": [{
                    "id": "1",
                    "type": "action",
                    "command": goal,
                    "purpose": "Execute direct instruction",
                    "depends_on": []
                }],
                "mental_model_updates": {"profile_updates": [], "new_rules": []}
            }