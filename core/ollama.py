import atexit
import json
import signal
import sys
import threading
import time
import requests
from typing import Dict, Any, Optional, List


class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434", default_timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.default_timeout = default_timeout
        self.current_loaded_model: Optional[str] = None

        self.role_models: Dict[str, str] = {
            "discovery": "qwen2.5-coder:7b",
            "decomposer": "qwen2.5-coder:7b",
            "planner": "qwen2.5-coder:14b",
            "critic": "qwen2.5-coder:14b",
            "embedding": "nomic-embed-text"
        }

        # Thread-safe exit hook for process termination
        atexit.register(self.unload_all)

        # Only bind OS signal handlers if initialized on the main Python thread
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except ValueError:
                pass  # Safely ignore if invoked within sub-thread wrappers
              
    def set_role_model(self, role: str, model_name: str) -> None:
        self.role_models[role] = model_name

    def get_role_model(self, role: str) -> str:
        return self.role_models.get(role, "qwen2.5-coder:7b")

    def list_available_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models_data = resp.json().get("models", [])
            return [m["name"] for m in models_data]
        except Exception as e:
            print(f"[Ollama] Warning: Could not fetch local models: {e}")
            return list(set(self.role_models.values()))

    def unload_model(self, model_name: str) -> bool:
        if not model_name:
            return True
        try:
            payload = {"model": model_name, "keep_alive": "0s"}
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=15
            )
            if model_name == self.current_loaded_model:
                self.current_loaded_model = None
            return resp.status_code == 200
        except Exception as e:
            print(f"[Ollama] Error unloading model '{model_name}': {e}")
            return False

    def unload_all(self) -> None:
        print("\n🛑 Cleaning up VRAM allocations...")
        for model in set(self.role_models.values()):
            self.unload_model(model)

    def _signal_handler(self, sig, frame) -> None:
        self.unload_all()
        sys.exit(130)

    def _ensure_single_model_in_vram(self, target_model: str) -> None:
        if self.current_loaded_model and self.current_loaded_model != target_model:
            print(f"🔄 Swapping VRAM: Unloading '{self.current_loaded_model}' -> Loading '{target_model}'")
            self.unload_model(self.current_loaded_model)
            time.sleep(1.0)

        self.current_loaded_model = target_model

    def generate(
        self, 
        role: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        stream: bool = True  # Enable streaming by default to avoid silent black-box hangs[cite: 3]
    ) -> str:
        model = self.get_role_model(role)
        self._ensure_single_model_in_vram(model)
  
        print(f"🤖 Calling '{role}' ({model})...")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": "5m",  # Keep warm in VRAM for fast subsequent calls[cite: 2]
            "options": options or {
                "temperature": 0.2,
                "num_ctx": 4096,
                # "num_predict": ('1.5' in model) and 512 or None,  # Prevent infinite generation loops[cite: 1]
                "stop": ["[END]", "User:", "\n\nUser"]
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            if stream:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout or self.default_timeout,
                    stream=True
                )
                resp.raise_for_status()
                full_response = []
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        content = chunk.get("response", "")
                        print(content, end="", flush=True)  # Print tokens live[cite: 3]
                        full_response.append(content)
                print()  # Newline
                return "".join(full_response).strip()
            else:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout or self.default_timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "").strip()

        except Exception as e:
            print(f"\n❌ Failure during execution with role '{role}' ({model}): {e}")
            self.unload_model(model)
            raise e
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the input text using the configured embedding model.
        """
        model = self.role_models.get("embedding", "nomic-embed-text")
        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=15
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("embedding", [])
            else:
                print(f"[Ollama Embedding] Returned status code {res.status_code}")
                return []
        except Exception as e:
            print(f"[Ollama Embedding] Failed to generate embedding: {e}")
            return []