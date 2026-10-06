# SysAgent

SysAgent is an elite, secure, and self-correcting Linux DevSecOps Orchestrator. It operates as an interactive Admin Advisor, breaking down high-level linguistic instructions into structured execution pipelines with long-term memory RAG layers. Instead of blindly executing commands, SysAgent safely probes the host environment to build a persistent mental model of the machine's hardware profile, software stack, and system quirks.

## Core Features

* **Self-Introspective State Mapping:** The agent dynamically categorizes system quirks and unexpected results discovered during operations.


* **Unified Rule Distillation:** The engine consolidates historical rules, raw CLI failures, and user feedback into concise, positive system directives prior to generating plans.


* **Phase Grouping & Execution:** Tasks are grouped logically into functional phases, such as Discovery/Inspection, Configuration/Setup, and Execution/Verification.


* **Interactive Phase Skipping:** If a phase stalls, the system offers human-in-the-loop gates to skip non-dependent phases while maintaining execution barriers for hard dependencies.


* **Tiered Decomposer-Worker Architecture:** Tasks are split by smaller models and executed by workers, reserving the 14B model strictly for deep diagnostic analysis during failures.



## Technology Stack

* **Backend Application:** Go (Golang) 1.22+.


* **Platform:** Linux, with targeted deployment on Debian/Ubuntu host environments.


* **Database Engine:** SQLite utilizing the modernc.org/sqlite pure-Go driver.


* **Inference Engine:** Local Ollama API daemon running specialized weights on a dedicated host GPU.


* **UI & Memory Abstraction (Planned):** Streamlit for an interactive chat and system state dashboard, and LlamaIndex for lazy-loading local documentation and help files.

## Database Architecture

The system utilizes a unified SQLite database (`agent_knowledge.db`) to maintain its persistent mental model. On startup, the core routine initializes the connection, migrates tables, and runs lightweight maintenance tasks. Key tables include:

* **`system_profile`:** Caches permanent environmental facts, preventing redundant probing for static constraints like hardware limitations or kernel versions.


* **`system_rules` / `system_quirks`:** Stores learned behaviors, context directives, and user feedback natively so the agent can modify its own operational state.


* **`telemetry_logs`:** Tracks historical task outputs and their high-dimensional vector embeddings for long-term semantic search injection.


* **`chat_sessions` & `session_history`:** Tracks high-level conversational workspaces to ensure multi-run continuity across execution threads.



## AI Models Utilized

* **Planner Model (14B):** `qwen2.5-coder:14b` handles structural execution schema generation.


* **Task Decomposer:** `qwen2.5-coder:7b` (or 1B-3B variants) is utilized to deterministically split user prompts without heavy hardware overhead.


* **Failure Interpreter (1.5B):** `qwen2.5-coder:1.5b` handles fast, precise error diagnosis and failure interpretation via structured text frames.


* **Embedding Model:** `nomic-embed-text` generates mathematical vector coordinates for local semantic RAG features.