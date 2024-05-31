import json
import hashlib

from typing import Dict, Any

def create_content_hash(data_dict: Dict[str, Any]) -> str:
        # Convert the dictionary to a JSON string
        json_str = json.dumps(data_dict, sort_keys=True)
        # Compute the SHA-256 hash
        content_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

        return f"flow-{content_hash}"