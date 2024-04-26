import os

import yaml
from huggingface_hub import HfApi

from flow_merge.lib.constants import ConfigKeys, MergeMethodIdentifier
from flow_merge.lib.logger import get_logger
from flow_merge.lib.merge_config import MergeConfig

logger = get_logger(__name__)

MODEL_CARD_TEMPLATE = """---
{metadata}
---
# {model_name}

This model is the result of merge of the following models made with flow-merge:

{models}

## flow-merge config

The following configuration was used to merge the models:

```yaml
{config}
```
"""


def generate_model_card(merge_config: MergeConfig, model_name: str = None):
    """
    Generates a model card for the model repository in HF hub.
    """
    # list of models
    task_arith_based_methods = [
        MergeMethodIdentifier.TIES_MERGING.value,
        MergeMethodIdentifier.DARE_TIES_MERGING.value,
        MergeMethodIdentifier.ADDITION_TASK_ARITHMETIC.value,
    ]

    if merge_config.data.method in task_arith_based_methods:
        base_model_name = merge_config.base_model.path
        model_path_list = [model.path for model in merge_config.models]
        # convert to text
        models = ""
        models += "- Base model:\n\t- " + base_model_name + "\n"
        models += "- Models:\n"
        for model in model_path_list:
            models += "\t- " + model + "\n"
    else:
        model_path_list = [
            model.path for model in [merge_config.base_model] + merge_config.models
        ]
        # convert to text
        models = "- Models:\n"
        for model in model_path_list:
            models += "\t- " + model + "\n"

    # tags
    tags = ["flow-merge", "merge"]

    config = merge_config.data.model_dump(exclude={'hf_hub_settings': {'token'}}, 
                                          exclude_unset=True)
    config["method"] = config["method"].value
    metadata = yaml.dump({"tags": tags, "library_name": "transformers"})

    fmt_card = MODEL_CARD_TEMPLATE.format(
        metadata=metadata,
        model_name=model_name,
        models=models,
        config=yaml.dump(config, indent=2, sort_keys=False, default_flow_style=False),
    )

    with open(
        os.path.join(merge_config.directory_settings.output_dir, "README.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(fmt_card)


def upload_model_to_hub(
    model_dir: str,
    username: str,
    model_name: str,
    private: bool = True,
    token: str = None,
):
    """
    This script uploads a model to the Hugging Face Hub.

    Args:
        model_dir: Path to the directory where the model is saved (output_dir in the merge config).
        username: Your username for the Hugging Face Hub.
        model_name: Name of the model.
        private: Whether the model should be private or public.
        token: Your authentication token for the Hugging Face Hub.

    Usage:
        upload_model_to_hub("/path/to/model", "my_username", "my_model", private=False, token="my_hf_token")
    """

    try:
        api = HfApi(token=token)

        repo_name = f"{username}/{model_name}"

        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
            private=private,
        )

        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            repo_type="model",
        )
    except Exception as e:
        raise RuntimeError(f"Error uploading model to Hugging Face Hub: {e}")
