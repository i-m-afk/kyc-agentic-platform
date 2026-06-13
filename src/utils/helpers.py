import os
import time
from typing import Dict, Any

def get_config(key: str, default: Any = None) -> Any:
    """Gets config from environment variables or Streamlit secrets fallback."""
    val = os.environ.get(key)
    if val is not None:
        if isinstance(default, bool):
            return val.lower() in ("true", "1", "yes")
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
        return val
    
    # Fallback to streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default

def get_mock_ml_flag() -> bool:
    return get_config("MOCK_ML", True)

def get_vllm_api_url() -> str:
    return get_config("VLLM_API_URL", "http://localhost:8000/v1")

def get_liveness_model_path() -> str:
    return get_config("LIVENESS_MODEL_PATH", "notebooks/liveness_model.pt")

def create_audit_entry(agent_name: str, status: str, latency: float, model: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "latency_seconds": round(latency, 3),
        "model_used": model,
        "details": details or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
