import argparse
import json
import os
import sys
from enum import Enum
from pathlib import Path
import yaml

from flow_merge.lib import di
from flow_merge.lib.config import ApplicationConfig, app_config
from flow_merge.lib.loaders.normalizer import NormalizationRunner
from flow_merge.lib.merge_config import MergeConfig
from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.merger.merger import Merger

class FileFormat(Enum):
    YAML = 'yaml'
    JSON = 'json'
    UNSUPPORTED = 'unsupported'


def detect_file_format(file_path: Path) -> FileFormat:
    try:
        with open(file_path, 'r') as file:
            # Try to load as JSON
            json.load(file)
            return FileFormat.JSON
    except json.JSONDecodeError:
        pass

    try:
        with open(file_path, 'r') as file:
            # Try to load as YAML
            yaml.safe_load(file)
            return FileFormat.YAML
    except yaml.YAMLError:
        pass

    return FileFormat.UNSUPPORTED


def is_merge_plan_file(path: Path) -> bool:
    if detect_file_format(path) != FileFormat.JSON:
        return False

    with open(path, 'r') as file:
        parsed = json.load(file)
        return (
                "created_at" in parsed and
                "lib_version" in parsed and
                "sha" in parsed
        )


def load_configuration_from_file(path: Path) -> MergeConfig:
    file_type = detect_file_format(path)
    if file_type == FileFormat.UNSUPPORTED:
        sys.exit("Provided file is not valid JSON or YAML. Ensure correct file was provided.")

    try:
        match file_type:
            case FileFormat.YAML:
                return MergeConfig.from_yaml(path)
            case FileFormat.JSON:
                return MergeConfig.from_json(path)
    except Exception as e:
        raise Exception("Failed to load configuration from file. Error: {}".format(e)) from e


def validate(args):
    path = Path(args.file)
    if not path.exists() or not path.is_file():
        sys.exit("Provided path does not exist or is not a file.")

    try:
        load_configuration_from_file(path)
    except Exception as e:
        sys.exit(str(e))


def run(args):
    path = Path(args.file)
    if not path.exists() or not path.is_file():
        sys.exit("Provided path does not exist or is not a file.")

    config = ApplicationConfig(
        device=args.device,
        hf_token=args.hf_token or os.getenv("HF_TOKEN"),
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )
    app_config.set(config)

    # Distinguish between plan files (json?) and merge configuration in JSON/YAML
    try:
        if is_merge_plan_file(path):
            merge_plan = MergePlan.from_file(path)
        else:
            config = load_configuration_from_file(path)
            merge_plan = MergePlan.from_config(config, normalization_runner=di.get(NormalizationRunner))
    except Exception as e:
        sys.exit(str(e))

    try:
        merger: Merger = di.get(Merger)
        merger.execute(merge_plan=merge_plan)
    except Exception as e:
        raise Exception("Unexpected error while merging") from e


def plan(args):
    # Validate if the provided config is even correct
    path = Path(args.file)
    if not path.exists() or not path.is_file():
        sys.exit("Provided path does not exist or is not a file.")

    app_config.set(ApplicationConfig(output_dir=""))

    output_path = os.path.abspath(args.output)
    # Check output directory/file
    if not os.access(os.path.dirname(output_path), os.W_OK):
        sys.exit("Provided output path does not exist or is not writable.")

    try:
        config = load_configuration_from_file(path)
        merge_plan = MergePlan.from_config(config, normalization_runner=di.get(NormalizationRunner))
        with(open(output_path, 'w')) as output_file:
            output_file.write(merge_plan.to_json())
    except Exception as e:
        raise e
        sys.exit(str(e))


def main():
    parser = argparse.ArgumentParser(description="Flow merge CLI")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate merge configuration')
    validate_parser.add_argument('file', type=str, help='Merge configuration file to validate')
    validate_parser.set_defaults(func=validate)

    # Run command
    run_parser = subparsers.add_parser('run', help='Run merge from merge configuration file or saved merging plan')
    run_parser.add_argument('file', type=str, help='Merge configuration file OR saved merging plan to run')
    run_parser.add_argument("--output-dir", "-o", required=True, type=str,
                            help="Output directory where the results will be saved")
    run_parser.add_argument("--device", required=False, default="cpu", type=str, help="PyTorch device to use")
    run_parser.add_argument("--hf-token", required=False, type=str,
                            help="HuggingFace access token. Alternatively HF_TOKEN environment variable can be used.")
    run_parser.add_argument("--local-dir", required=False, type=str, default="models",
                            help="Working directory for the library. Models and their configurations will be downloaded there.")
    run_parser.add_argument("--trust-remote-code", required=False, default=False, type=bool,
                            help="Whether to trust remote code execution (see HuggingFace documentation for details).")
    run_parser.set_defaults(func=run)

    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Prepare merge plan from provided merge configuration')
    plan_parser.add_argument('file', type=str, help='Input merge configuration file')
    plan_parser.add_argument('--output', '-o', required=True, type=str,
                             help='Output merge plan file to save the merge plan to')
    plan_parser.set_defaults(func=plan)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
