import streamlit as st
import subprocess
from core.advisor import AdminAdvisor

# Page configuration
st.set_page_config(
    page_title="SysAgent Admin Advisor",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SysAgent: Admin Advisor Console")

# Initialize Advisor in Session State
if "advisor" not in st.session_state:
    st.session_state.advisor = AdminAdvisor()

advisor = st.session_state.advisor

# -----------------------------------------------------------------------------
# SIDEBAR: Model Management & System Profile
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Role Assignments")
    available_models = advisor.ollama.list_available_models()

    if available_models:
        discovery_m = st.selectbox(
            "Discovery Model", 
            available_models, 
            index=0 if "qwen2.5-coder:7b" in available_models else 0
        )
        planner_m = st.selectbox(
            "Planner Model", 
            available_models, 
            index=available_models.index("qwen2.5-coder:14b") if "qwen2.5-coder:14b" in available_models else 0
        )

        advisor.ollama.set_role_model("discovery", discovery_m)
        advisor.ollama.set_role_model("decomposer", discovery_m)
        advisor.ollama.set_role_model("planner", planner_m)
    else:
        st.error("No Ollama models detected. Ensure Ollama daemon is running.")

    st.divider()
    st.header("🧠 Vector Memory Index")
    memory_count = advisor.memory.get_memory_count()
    st.caption(f"**Stored Execution Records:** `{memory_count}`")

    st.divider()
    st.header("💻 Cached System Profile")
    profile = advisor.db.get_system_profile()
    if profile:
        for p in profile:
            st.caption(f"**{p['category']}.{p['attribute']}**: `{p['value']}`")
    else:
        st.info("No static profile facts cached yet.")

    st.divider()
    st.header("📌 Learned Rules")
    rules = advisor.db.get_system_rules()
    if rules:
        for r in rules:
            st.caption(f"• **[{r['context_key']}]**: {r['description']}")
    else:
        st.info("No system rules recorded.")

# -----------------------------------------------------------------------------
# MAIN VIEW: Workspace & Editable Script Sandbox
# -----------------------------------------------------------------------------
user_prompt = st.text_input(
    "Enter an administration goal or task:",
    placeholder="e.g., Audit open network ports and allow incoming SSH traffic"
)

if st.button("Generate Strategy & Script", type="primary"):
    if user_prompt.strip():
        with st.spinner("Analyzing environment, distilling rules, and planning..."):
            result = advisor.analyze_and_plan(user_prompt)
            st.session_state.current_result = result
    else:
        st.warning("Please enter a valid command or goal.")

# Display Results if Available
if "current_result" in st.session_state:
    res = st.session_state.current_result

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Task Strategy & Steps")
        st.write(f"**Planner Strategy:** {res['plan'].get('reasoning', 'N/A')}")

        if res.get("historical_memories"):
            st.markdown("**Retrieved Vector Memories:**")
            for mem in res["historical_memories"]:
                st.caption(f"```\n{mem[:150]}...\n```")

        st.markdown("**Distilled Directives:**")
        for rule in res["distilled_rules"]:
            st.info(rule)

    with col2:
        st.subheader("📝 Editable Action Script")
        st.caption("Review or edit the generated script before execution:")

        # Editable code editor block
        edited_script = st.text_area(
            "Script Output",
            value=res["generated_script"],
            height=300
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🚀 Execute Approved Script"):
                st.info("Executing script on host system...")
                try:
                    process = subprocess.run(
                        edited_script,
                        shell=True,
                        capture_output=True,
                        text=True,
                        executable="/bin/bash"
                    )
                    if process.returncode == 0:
                        st.success("Execution Successful!")
                        output_text = process.stdout if process.stdout else "Command completed with no output."
                        st.code(output_text)

                        # Persist successful run into ChromaDB
                        advisor.record_successful_execution(
                            goal=user_prompt,
                            plan_summary=res['plan'].get('reasoning', ''),
                            output=output_text
                        )
                        st.toast("Saved execution pattern to Vector Memory!", icon="🧠")
                    else:
                        st.error(f"Execution Failed (Exit Code {process.returncode})")
                        st.code(process.stderr)

                        # Trigger Critic Failure Diagnosis
                        with st.spinner("🤖 Critic is analyzing execution failure and distilling system rule..."):
                            diagnosis = advisor.diagnose_execution_failure(
                                command=edited_script,
                                error_output=process.stderr if process.stderr else process.stdout,
                                exit_code=process.returncode
                            )

                        # Safely parse diagnosis dictionary
                        if diagnosis and isinstance(diagnosis, dict):
                            st.markdown("---")
                            st.subheader("🔍 Critic Diagnosis & Learned Directive")
                            st.warning(diagnosis.get("explanation", "No diagnosis output produced."))

                            extracted = diagnosis.get("extracted_rule")
                            if extracted:
                                if diagnosis.get("rule_saved"):
                                    st.success(f"🧠 **Learned New System Rule:** `{extracted}`")
                                    st.rerun()
                                else:
                                    st.info(f"ℹ️ **Rule Already Exists in Database:** `{extracted}`")
                        else:
                            st.error("Stale session detected. Re-initializing Advisor...")
                            st.session_state.advisor = AdminAdvisor()
                            st.rerun()

                except Exception as e:
                    st.error(f"Error launching process: {e}")

        with btn_col2:
            if st.button("🗑️ Clear Workspace"):
                del st.session_state.current_result
                st.rerun()