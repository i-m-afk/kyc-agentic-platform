import base64
import json
import httpx
from datetime import date
from src.schemas.models import ExtractionResult
from src.utils.helpers import get_mock_ml_flag, get_vllm_api_url

def extract_document_info(image_path: str) -> ExtractionResult:
    """
    Extracts key details (Name, DOB, ID number) from the uploaded ID card image.
    Uses local Mock mode or vLLM vision model inference.
    """
    # 1. Check if we should use local Mock mode
    if get_mock_ml_flag():
        filename = image_path.lower()
        if "jane" in filename:
            return ExtractionResult(
                name="Jane Doe",
                dob=date(1990, 5, 15),
                id_number="JD9900515",
                confidence=0.98
            )
        elif "john" in filename:
            return ExtractionResult(
                name="John Doe",
                dob=date(1985, 11, 23),
                id_number="JD851123X",
                confidence=0.99
            )
        elif "robert" in filename:
            return ExtractionResult(
                name="Robert Vance",
                dob=date(1978, 2, 14),
                id_number="RV780214",
                confidence=0.95
            )
        else:
            return ExtractionResult(
                name="Alice Smith",
                dob=date(1995, 8, 30),
                id_number="AS950830",
                confidence=0.90
            )

    # 2. Actual vLLM Vision inference
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read image file for document extraction: {str(e)}")

    api_url = get_vllm_api_url()
    headers = {"Content-Type": "application/json"}
    
    prompt = (
        "Extract the following fields from this ID card image: "
        "1. name (full name as string) "
        "2. dob (date of birth in YYYY-MM-DD format) "
        "3. id_number (document reference number) "
        "Return ONLY a valid JSON object matching the schema: "
        '{"name": "...", "dob": "YYYY-MM-DD", "id_number": "...", "confidence": 0.95}. '
        "Do not include any markdown fences or additional explanation."
    )

    payload = {
        "model": "qwen2-vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }

    try:
        response = httpx.post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"].strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        dob_parts = [int(p) for p in data["dob"].split("-")]
        
        return ExtractionResult(
            name=data["name"],
            dob=date(dob_parts[0], dob_parts[1], dob_parts[2]),
            id_number=data["id_number"],
            confidence=data.get("confidence", 0.90)
        )
    except Exception as e:
        raise RuntimeError(f"vLLM Document Extraction failed: {str(e)}")
