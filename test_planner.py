
from core.ollama import OllamaManager
from core.db import DatabaseStore
from core.decomposer import TaskDecomposer
from core.distiller import RuleDistiller
from core.planner import ExecutionPlanner

# 1. Initialize Components
ollama = OllamaManager()
db = DatabaseStore()
decomposer = TaskDecomposer(ollama)
distiller = RuleDistiller(ollama)
planner = ExecutionPlanner(ollama)

# 2. Seed test profile/rule facts into SQLite
db.set_profile_fact("os", "distro", "Ubuntu 22.04")
db.add_rule("permissions", "[Permissions]: Use sudo when configuring UFW rules")

user_goal = "Audit open ports and allow inbound SSH traffic through the firewall."

print("1️⃣ Decomposing Goal...")
decomposition = decomposer.decompose_goal(user_goal)
print(f"Goal Breakdown: {decomposition}\n")

print("2️⃣ Distilling Rules...")
profile_facts = db.get_system_profile()
stored_rules = [f"[{r['context_key']}]: {r['description']}" for r in db.get_system_rules()]

distilled = distiller.distill_rules(
    phase_objective=user_goal,
    historical_rules=stored_rules
)
print(f"Distilled Directives: {distilled}\n")

print("3️⃣ Generating Execution Plan via Planner (14B)...")
plan = planner.generate_plan(
    goal=user_goal,
    profile_facts=profile_facts,
    distilled_rules=distilled,
    historical_memories=[]
)

print("\n📋 Execution Plan Output:")
print(f"Reasoning: {plan.get('reasoning')}\n")
for step in plan.get("steps", []):
    print(f"  🏷️ Step [{step['id']}] ({step['type'].upper()}): `{step['command']}`")
    print(f"     ↳ Purpose: {step['purpose']}")
    if step.get("depends_on"):
        print(f"     ↳ Depends On: {step['depends_on']}")
    print()
    