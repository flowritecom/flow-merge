import argparse
import json

import yaml

from flow_merge.lib.config import config
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import ValidatedInputData
from flow_merge.lib.merge_runner import run_merge
from flow_merge.lib.utils import upload_model_to_hub

logger = get_logger(__name__)


def print_schema():
    schema = ValidatedInputData.model_json_schema()
    print(json.dumps(schema, indent=2))


def print_valid_inputs():
    input_descriptions = {
        "Required parameters": {
            "base_model": "The base model to be used for merging",
            "models": [
                "List of dictionaries, each representing a model to be merged",
                "- model: Each model dictionary should have a 'model' property specifying the model path or identifier",
                "- weight: The 'weight' property in a model dictionary is optional and specifies the weight of the model during merging",
            ],
            "method": "The merge method to be used, one of ['addition-task-arithmetic', 'ties-merging', 'slerp', 'dare-ties-merging', 'model-soup', 'passthrough']",
        },
        "Optional parameters": {
            "device": "The device to be used for merging, one of ['cpu', 'cuda']",
            "method_global_parameters": [
                "Global parameters for the merge method",
                "- normalize: bool",
                "- p: float",
                "- scaling_coefficient: float",
                "- t: float",
                "- top_k: float",
            ],
            "directory_settings": [
                "Directories for caching, loading, and saving models",
                "- cache_dir: str",
                "- local_dir: str",
                "- output_dir: str",
            ],
            "hf_hub_settings": [
                "Settings for interacting with the Hugging Face Hub",
                "- token: str",
                "- trust_remote_code: bool",
            ],
            "tokenizer_settings": [
                "Settings for the tokenizer used with the merged model",
                "- interpolation_method: str",
                "- mode: str",
            ],
        },
    }

    for section, items in input_descriptions.items():
        print(f"\n# {section}")
        for key, description in items.items():
            if isinstance(description, list):
                print(f"- {key}:")
                for desc in description:
                    print(f"  {desc}")
            else:
                print(f"- {key}: {description}")


def validate_config(args):
    try:
        with open(args.config, "r") as file:
            config_data = file.read()
        config_dict = yaml.safe_load(config_data)
        ValidatedInputData.model_validate(config_dict)

        print("Configuration file is valid.")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
    except Exception as e:
        print(f"Configuration file is invalid: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process a YAML config")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Schema command
    schema_parser = subparsers.add_parser(
        "schema", help="Display the schema for the YAML config"
    )
    inputs_parser = subparsers.add_parser(
        "inputs", help="Display the valid input values for the YAML config"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a YAML config file"
    )
    validate_parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )

    # Merge command
    merge_parser = subparsers.add_parser("run", help="Run the model merging process")
    merge_parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )
    merge_parser.add_argument(
        "--model_name", type=str, help="Name of the resulting model", required=False
    )
    merge_parser.add_argument(
        "--token",
        default=config.hf_token,
        type=str,
        help="Token for the HF hub",
        required=False,
    )

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a model to the HF hub")
    upload_parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the directory where the model is save - output_dir in the merge config",
        required=True,
    )
    upload_parser.add_argument(
        "--username", type=str, help="Username for the HF hub", required=True
    )
    upload_parser.add_argument(
        "--model_name", type=str, help="Name of the model", required=True
    )
    upload_parser.add_argument(
        "--private",
        type=bool,
        default=True,
        help="Whether the model should be private or public",
        required=False,
    )
    upload_parser.add_argument(
        "--token",
        default=config.hf_token,
        type=str,
        help="Token for the HF hub",
        required=False,
    )
    args = parser.parse_args()

    if args.token:
        config.set_hf_token(args.token)

    if args.command == "run":
        run_merge(args)
    elif args.command == "schema":
        print_schema()
    elif args.command == "inputs":
        print_valid_inputs()
    elif args.command == "validate":
        validate_config(args)
    elif args.command == "upload":
        upload_model_to_hub(
            args.model_dir, args.username, args.model_name, args.private, args.token
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
