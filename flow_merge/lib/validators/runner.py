import asyncio
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, ValidationError

from flow_merge.lib.validators._directory import DirectorySettings
from flow_merge.lib.validators._method import MergeMethodSettings
from flow_merge.lib.validators._tokenizer import TokenizerSettings


async def validate_settings(base_model: Type[BaseModel], data: Optional[Dict[str, Any]]) -> Optional[BaseModel]:
    if data is None:
        return None
    try:
        return base_model(**data)
    except ValidationError as e:
        print(f"{base_model.__name__} validation error: {e}")
        return None


required_fields: Dict[str, Dict[str, Any]] = {
    "method": {"base_model": MergeMethodSettings, "required": True},
    "method_global_parameters": {"base_model": MergeMethodSettings, "required": False},
    "directory_settings": {"base_model": DirectorySettings, "required": False},
    "tokenizer_settings": {"base_model": TokenizerSettings, "required": False}
}

def runner(input_data: Dict[str, Any], required_fields: Dict[str, Dict[str, Any]] = required_fields):
    validated_data = {}
    for field, config in required_fields.items():
        base_model = config["base_model"]
        is_required = config["required"]
        data = input_data.get(field)
        if data is None and is_required:
            raise ValueError(f"The field '{field}' is required but not provided in the input data.")
        result = validate_settings(base_model, data)
        validated_data[field] = result

        if result:
            print(f"{field} validated successfully.")
        elif required_fields[field]["required"]:
            print(f"{field} validation failed.")

    return validated_data

async def async_runner(input_data: Dict[str, Any], required_fields: Dict[str, Dict[str, Any]] = required_fields):
    tasks = []
    for field, config in required_fields.items():
        base_model = config["base_model"]
        is_required = config["required"]
        data = input_data.get(field)
        if data is None and is_required:
            raise ValueError(f"The field '{field}' is required but not provided in the input data.")
        tasks.append(validate_settings(base_model, data))

    results = await asyncio.gather(*tasks)
    validated_data = {field: result for field, result in zip(required_fields.keys(), results)}

    for field, result in validated_data.items():
        if result:
            print(f"{field} validated successfully.")
        elif required_fields[field]["required"]:
            print(f"{field} validation failed.")

    return validated_data
