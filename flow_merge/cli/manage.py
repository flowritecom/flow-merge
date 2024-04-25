import argparse
import json

from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import ValidatedInputData
from flow_merge.lib.merge_runner import run_merge
from flow_merge.lib.utils import upload_model_to_hub

logger = get_logger(__name__)


def print_schema():
    schema = ValidatedInputData.model_json_schema()
    print(json.dumps(schema, indent=2))

def print_valid_inputs():
    print("\n# Required parameters")
    print("- 'base_model': \t\t\t the base model to be used for merging")
    print(
        "- 'models': \t\t\t\t list of dictionaries, each representing a model to be merged"
    )
    print(
        " \t- 'model': \t\t\t\t each model dictionary should have a 'model' property specifying the model path or identifier"
    )
    print(
        " \t- 'weight': \t\t\t\t the 'weight' property in a model dictionary is optional and specifies the weight of the model during merging"
    )
    print(
        "- 'method': \t\t\t\t the merge method to be used, one of ['addition-task-arithmetic','ties-merging','slerp','dare-ties-merging','model-soup','passthrough']"
    )
    print("\n# Optional parameters")
    print(
        "- 'device': \t\t\t\t the device to be used for merging one of ['cpu','cuda']"
    )
    print("- 'method_global_parameters': \t\t global parameters for the merge method")
    print(" \t- 'normalize': bool\t\t\t\t lorem ipsum")
    print(" \t- 'p': float\t\t\t\t\t lorem ipsum")
    print(" \t- 'scaling_coefficient': float\t\t lorem ipsum")
    print(" \t- 't': float\t\t\t\t\t lorem ipsum")
    print(" \t- 'top_k': float\t\t\t\t lorem ipsum")
    print(
        "- 'directory_settings': \t\t directories for caching, loading, and saving models"
    )
    print(" \t- 'cache_dir': str\t\t\t\t lorem ipsum")
    print(" \t- 'local_dir': str\t\t\t\t\t lorem ipsum")
    print(" \t- 'output_dir': str\t\t lorem ipsum")
    print(
        "- 'hf_hub_settings': \t\t\t settings for interacting with the Hugging Face Hub"
    )
    print(" \t- 'token': str\t\t\t\t lorem ipsum")
    print(" \t- 'trust_remote_code': bool\t\t\t\t\t lorem ipsum")
    print(
        "- 'tokenizer_settings': \t\t settings for the tokenizer used with the merged model"
    )
    print(" \t- 'interpolation_method': str\t\t lorem ipsum")
    print(" \t- 'mode': str\t\t\t\t lorem ipsum")


def validate_config(args):
    try:
        with open(args.config, "r") as file:
            config_data = file.read()

        # Load the config data as YAML
        import yaml

        config_dict = yaml.safe_load(config_data)

        # Validate the config data against the schema
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
    
    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload a model to the HF hub"
    )
    upload_parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the directory where the model is save - output_dir in the merge config",
        required=True
    )
    upload_parser.add_argument("--username", type=str, help="Username for the HF hub", required=True)
    upload_parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
    upload_parser.add_argument(
        "--private",
        type=bool,
        default=True,
        help="Whether the model should be private or public",
        required=False
    )
    upload_parser.add_argument("--token", type=str, help="Token for the HF hub", required=True)

    args = parser.parse_args()

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
