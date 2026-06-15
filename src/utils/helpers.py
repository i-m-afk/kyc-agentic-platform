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

def download_liveness_model(destination_path: str) -> bool:
    """Attempts to download the liveness model weights from GitHub releases."""
    import subprocess
    import urllib.request
    import urllib.error
    import json

    url = "https://github.com/i-m-afk/kyc-agentic-platform/releases/download/v1.0.0/liveness_model.pt"
    print(f"Liveness model not found at {destination_path}. Attempting to download from {url}...")

    # 1. Try downloading using gh CLI if available and authenticated
    try:
        cmd = [
            "gh", "release", "download", "v1.0.0",
            "--repo", "i-m-afk/kyc-agentic-platform",
            "-p", "liveness_model.pt",
            "-O", destination_path,
            "--clobber"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(destination_path):
            print("Successfully downloaded liveness_model.pt using gh CLI.")
            return True
        else:
            print(f"gh CLI download failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"gh CLI download attempt failed: {e}")

    # 2. Try downloading using Python urllib (with GITHUB_TOKEN fallback or direct url)
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"User-Agent": "KYC-Agentic-Platform-Downloader"}

    try:
        if token:
            print("GITHUB_TOKEN found. Attempting authenticated download via GitHub API...")
            headers["Authorization"] = f"token {token}"
            
            # Get the asset ID for v1.0.0
            api_url = "https://api.github.com/repos/i-m-afk/kyc-agentic-platform/releases/tags/v1.0.0"
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                release_data = json.loads(response.read().decode())
                
            asset_id = None
            for asset in release_data.get("assets", []):
                if asset.get("name") == "liveness_model.pt":
                    asset_id = asset.get("id")
                    break
                    
            if asset_id:
                # Download the asset using the octet-stream header
                asset_url = f"https://api.github.com/repos/i-m-afk/kyc-agentic-platform/releases/assets/{asset_id}"
                headers["Accept"] = "application/octet-stream"
                req_asset = urllib.request.Request(asset_url, headers=headers)
                with urllib.request.urlopen(req_asset) as response, open(destination_path, "wb") as out_file:
                    out_file.write(response.read())
                print("Successfully downloaded liveness_model.pt via GitHub API.")
                return True
            else:
                print("Could not find liveness_model.pt asset in the release.")
        else:
            # Direct download fallback
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(destination_path, "wb") as out_file:
                out_file.write(response.read())
            print("Successfully downloaded liveness_model.pt directly.")
            return True
            
    except Exception as e:
        print(f"Python urllib download failed: {e}")

    print(f"WARNING: Could not download liveness_model.pt automatically. "
          f"Please manually download the weights from GitHub Releases and place them at {destination_path}.")
    return False

def get_liveness_model_path() -> str:
    default_path = "notebooks/liveness_model.pt"
    if os.path.basename(os.getcwd()) == "notebooks":
        default_path = "liveness_model.pt"
        
    path = get_config("LIVENESS_MODEL_PATH", default_path)
    if os.path.exists(path):
        return path
    if os.path.exists("liveness_model.pt") and os.path.basename(os.getcwd()) != "notebooks":
        return "liveness_model.pt"
    
    # Neither exists, attempt to download to the target path
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    download_liveness_model(path)
    return path

def create_audit_entry(agent_name: str, status: str, latency: float, model: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "latency_seconds": round(latency, 3),
        "model_used": model,
        "details": details or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
