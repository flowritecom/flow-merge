import os
import re
from pydantic import BaseModel, Field, field_validator, ValidationError
import logging
from huggingface_hub import login, logout

from flow_merge.lib.constants import DeviceIdentifier

logger = logging.getLogger(__name__)


class ApplicationConfig(BaseModel):
    device: DeviceIdentifier = Field(default=os.getenv("DEVICE", "cpu"))
    hf_token: str = Field(..., default_factory=lambda: os.getenv("HF_TOKEN"))

    def __post_init__(self):
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        logger.info("HF_HUB_DISABLE_IMPLICIT_TOKEN set to 1 to disable implicit token authentication.")

    def set_hf_token(self, token: str):
        self.hf_token = token

    def set_device(self, device: str):
        self.device = device


    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v is not None and v not in ["cpu", "cuda"]:
            raise ValidationError(
                "device",
                f"Invalid device: {v}. Supported devices are 'cpu' and 'cuda'.",
            )
        return v

    @field_validator("hf_token")
    def validate_hf_token(cls, v):
        if v:
            token_pattern = r"^hf_[a-zA-Z0-9]+$"
            if not re.match(token_pattern, v):
                logger.warning(
                    f"Invalid Hugging Face Hub token format. HF token should be of the form '{token_pattern}'."
                )
                raise ValueError("Invalid token format")

            try:
                login(token=v)
                logout(token=v)
            except Exception as e:
                logger.warning(
                    f"Failed to login to the Hugging Face Hub with the provided token: {e}"
                )
                raise ValueError("Failed to authenticate with the provided token")
        return v
