from core.ollama import OllamaManager

ollama = OllamaManager()

# 1. Dynamically re-assign planner role to a smaller model if needed
# ollama.set_role_model("planner", "qwen2.5-coder:7b")

# 2. Run discovery phase (uses 1.5B model)
discovery_output = ollama.generate(
    role="discovery", 
    prompt="Extract read-only options for ufw from the help output: ..."
)

print("Discovery Output:", discovery_output)

# 3. Run planning phase (automatically unloads 1.5B, pauses 1s, loads 7B model)
plan_output = ollama.generate(
    role="planner", 
    prompt="Generate task graph for: Check open firewall ports."
)
print("Plan Output:", plan_output)
