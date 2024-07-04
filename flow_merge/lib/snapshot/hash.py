import json
import hashlib

from typing import Dict, Any, Union

def create_content_hash(data: Union[str, Dict[str, Any]], is_json=False) -> str:
        # Convert the dictionary to a JSON string
        if not is_json:
                data = json.dumps(data, sort_keys=True)
        
        # Compute the SHA-256 hash
        content_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()

        return f"flow-{content_hash}"