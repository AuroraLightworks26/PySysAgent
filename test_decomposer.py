from core.ollama import OllamaManager
from core.decomposer import TaskDecomposer
from core.distiller import RuleDistiller

ollama = OllamaManager()
decomposer = TaskDecomposer(ollama)
distiller = RuleDistiller(ollama)

# 1. Test Decomposition
plan = decomposer.decompose_goal("Audit active SSH connections and restrict port 22 access using UFW")
print("📋 Decomposed Plan:", plan)

# 2. Test Rule Distillation with raw CLI failure
rules = distiller.distill_rules(
    phase_objective="Configure firewall rules",
    raw_failure="ufw: command not found, permission denied writing to /etc/ufw/",
    historical_rules=["[Permissions]: Use sudo for package installation and system files"]
)
print("📌 Distilled Directives:", rules)