
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

import torch

class HardwareSettings(BaseModel, arbitrary_types_allowed=True):
    device: Optional[torch.device] = Field(
        # FIXME set auto-recognize 
        default = "cpu",
        description=str("The device to use for tensor operations." +
        "Defaults to 'cuda' if available, otherwise 'cpu'.")
    )

    def _unpack(self):
        return self.device

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v is not None and v not in ["cpu", "cuda"]:
            raise ValidationError(
                "device",
                f"Invalid device: {v}. Supported devices are 'cpu' and 'cuda'.",
            )
        return v
