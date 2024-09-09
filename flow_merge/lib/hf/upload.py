
import os
import yaml
import logging

from huggingface_hub import HfApi

from flow_merge.lib.merge_plan import MergePlan
from flow_merge.lib.config import ApplicationConfig

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
{metadata}
---
# {model_name}

## flow-merge config

The following configuration was used to merge the models:

```yaml
{config}
```
"""


def generate_model_card(
        merge_plan: MergePlan, 
        app_config: ApplicationConfig,
        model_name: str = None
    ):
    """
    Generates a model card for the model repository in HF hub.
    """
    # tags
    tags = ["flow-merge", "merge"]

    metadata = yaml.dump({"tags": tags, "library_name": "transformers"})

    fmt_card = MODEL_CARD_TEMPLATE.format(
        metadata=metadata,
        model_name=model_name,
        config=yaml.dump(
            merge_plan.raw_config, 
            indent=2, 
            sort_keys=False, 
            default_flow_style=False
        ),
    )

    with open(
        os.path.join(app_config.output_dir, "README.md"),
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
