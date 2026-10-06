import re
import subprocess
from typing import Dict, Any
from core.ollama import OllamaManager
from core.db import DatabaseStore


class ExecutionCritic:
    def __init__(self, ollama: OllamaManager, db: DatabaseStore):
        self.ollama = ollama
        self.db = db

    def _fetch_targeted_docs(self, command: str, error_msg: str) -> str:
        tokens = command.strip().split()
        if not tokens:
            return ""

        base_cmd = tokens[0]
        if base_cmd in ["sudo", "cd", "export"] and len(tokens) > 1:
            base_cmd = tokens[1]

        doc_output = ""
        try:
            res = subprocess.run(
                [base_cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=2
            )
            doc_output = res.stdout if res.stdout else res.stderr
        except Exception:
            pass

        if not doc_output or len(doc_output.strip()) < 30:
            try:
                res = subprocess.run(
                    f"man {base_cmd} | col -b",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0:
                    doc_output = res.stdout
            except Exception:
                pass

        if not doc_output or not doc_output.strip():
            return ""

        flag_match = re.search(r"--?[a-zA-Z0-9-]+", error_msg)
        if flag_match:
            flag = flag_match.group(0)
            lines = doc_output.splitlines()
            focused = []
            for i, line in enumerate(lines):
                if flag in line:
                    start = max(0, i - 1)
                    end = min(len(lines), i + 3)
                    focused.extend(lines[start:end])
            if focused:
                return f"=== Targeted Doc Snippet for '{flag}' ===\n" + "\n".join(focused[:15])

        lines = doc_output.splitlines()[:25]
        return f"=== Command Usage Snippet ({base_cmd}) ===\n" + "\n".join(lines)

    def analyze_failure(self, command: str, error_output: str, exit_code: int = 1) -> Dict[str, Any]:
        explanation = "Unable to generate diagnosis."
        extracted_rule = ""
        saved = False

        try:
            doc_snippet = self._fetch_targeted_docs(command, error_output)
            context_block = f"Failed Command/Script: {command}\nExit Code: {exit_code}\nError Output:\n{error_output}"
            if doc_snippet:
                context_block += f"\n\n{doc_snippet}"

            system_prompt = (
                "You are an expert Linux System Administration Critic and Failure Analyst.\n"
                "Analyze command execution failures and explain the root cause concisely.\n"
                "Provide clear, actionable troubleshooting advice."
            )

            user_prompt = (
                f"=== EXECUTION FAILURE DETAILS ===\n"
                f"{context_block}\n\n"
                f"Explain why this failed and provide the exact corrected command or strategy."
            )

            raw_explanation = self.ollama.generate(
                role="critic",
                prompt=user_prompt,
                system_prompt=system_prompt,
                options={"temperature": 0.2}
            )
            if raw_explanation:
                explanation = raw_explanation.strip()

            rule_prompt = (
                "Based on the following diagnostic analysis, synthesize ONE concise, positive system rule for future agent planning.\n\n"
                "CRITICAL RULES FOR OUTPUT:\n"
                "1. Format strictly as: [Domain/Topic]: Positive Actionable Directive\n"
                "2. State what TO DO rather than what NOT to do.\n"
                "3. Keep it under 15 words.\n"
                "4. Do NOT output conversational text—ONLY the formatted rule line.\n\n"
                f"Diagnostic Analysis:\n{explanation}\n\n"
                f"Canonical System Rule:"
            )

            rule_resp = self.ollama.generate(
                role="decomposer",
                prompt=rule_prompt,
                options={"temperature": 0.1, "stop": ["\n"]}
            )
            if rule_resp:
                extracted_rule = rule_resp.strip().strip('"\'`')

            # Parse and save rule safely
            if extracted_rule:
                context_key = "failure_learning"
                match = re.match(r"^:\s*(.*)$", extracted_rule)
                if match:
                  context_key = match.group(1).lower().strip()
                  saved = self.db.save_system_rule(context_key, extracted_rule)

        except Exception as e:
            print(f"[Critic Error] {e}")
            explanation = f"Critic diagnosis error: {e}"

        return {
            "explanation": explanation,
            "extracted_rule": extracted_rule,
            "rule_saved": saved
        }