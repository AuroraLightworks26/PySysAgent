import re
from typing import List, Optional, Dict
from core.ollama import OllamaManager


class RuleDistiller:
    def __init__(self, ollama: OllamaManager):
        self.ollama = ollama

    def prune_irrelevant_rules(self, objective: str, rules: List[str]) -> List[str]:
        """Filters stored historical rules based on key domain matches in the objective[cite: 2]."""
        if not rules:
            return []

        obj_lower = objective.lower()
        relevant = []

        for rule in rules:
            rule_lower = rule.lower()
            is_relevant = False

            if "[permissions]" in rule_lower or "sudo" in rule_lower:
                is_relevant = any(k in obj_lower for k in ["permission", "sudo", "install", "write", "/etc", "service", "systemd"])
            elif "[services]" in rule_lower or "systemctl" in rule_lower:
                is_relevant = any(k in obj_lower for k in ["service", "systemd", "daemon", "status", "start", "restart"])
            elif "[packages]" in rule_lower or "apt" in rule_lower:
                is_relevant = any(k in obj_lower for k in ["package", "install", "apt", "dep"])
            elif "[network]" in rule_lower or "curl" in rule_lower or "port" in rule_lower:
                is_relevant = any(k in obj_lower for k in ["network", "port", "http", "nginx", "curl", "ip"])
            else:
                is_relevant = True  # Retain general rules

            if is_relevant:
                relevant.append(rule)

        return relevant

    def distill_rules(
        self,
        phase_objective: str,
        raw_failure: str = "",
        doc_snippets: str = "",
        historical_rules: Optional[List[str]] = None,
        user_feedback: str = ""
    ) -> List[str]:
        """
        Consolidates multi-source diagnostics and historical context into 1-3 
        task-scoped positive directives[cite: 2].
        """
        historical = historical_rules or []
        pruned_historical = self.prune_irrelevant_rules(phase_objective, historical)

        # Fast path: If context is clean and historical pool is minimal, avoid LLM call[cite: 2]
        if not raw_failure and not doc_snippets and not user_feedback and len(pruned_historical) <= 3:
            if pruned_historical:
                return pruned_historical

        context_blocks = [f"=== CURRENT PHASE OBJECTIVE ===\n{phase_objective}\n"]

        if pruned_historical:
            context_blocks.append("=== HISTORICAL SYSTEM RULES ===")
            for r in pruned_historical:
                context_blocks.append(f"- {r}")
            context_blocks.append("")

        if user_feedback:
            context_blocks.append(f"=== USER CORRECTION & FEEDBACK ===\n{user_feedback}\n")

        if raw_failure:
            context_blocks.append(f"=== RECENT CLI FAILURE DIAGNOSTIC ===\n{raw_failure}\n")

        if doc_snippets:
            context_blocks.append(f"=== REFERENCE MANUAL SNIPPETS ===\n{doc_snippets}\n")

        raw_inputs = "\n".join(context_blocks)

        system_prompt = (
            "You are a System Administration Context Distiller.\n"
            "Your job is to synthesize raw rules, diagnostics, documentation, and feedback into 1 to 3 sharp, "
            "positive, task-scoped execution directives for the current phase objective.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output MUST be between 1 and 3 bullet points maximum.\n"
            "2. Format each rule strictly as: [Domain]: Actionable Positive Directive\n"
            "   Examples:\n"
            "   - [Permissions]: Use 'sudo' when modifying files in /etc/nginx/\n"
            "   - [Services]: Query authentication logs using 'journalctl -u ssh' instead of log files\n"
            "3. Focus strictly on directives relevant to the current phase objective.\n"
            "4. Do NOT output explanations, conversational text, or code blocks."
        )

        prompt = f"Raw Inputs:\n{raw_inputs}\n[END]"

        try:
            resp = self.ollama.generate(
                role="decomposer",
                prompt=prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                    "stop": ["[END]", "User:"]
                }
            )
            distilled = self._parse_distilled_rules(resp)
            return distilled if distilled else pruned_historical

        except Exception as e:
            print(f"[Distiller] Warning: Distillation pass failed ({e}), using pruned historical rules.")
            return pruned_historical

    def _parse_distilled_rules(self, response: str) -> List[str]:
        """Extracts strictly formatted [Domain]: Directive lines[cite: 2]."""
        rules = []
        for line in response.splitlines():
            trimmed = line.strip()
            trimmed = re.sub(r"^[-*\d.]+\s*", "", trimmed)
            if trimmed.startswith("[") and "]:" in trimmed:
                rules.append(trimmed)
                if len(rules) >= 3:
                    break
        return rules