
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

from flow_merge.lib.constants import DeviceIdentifier

class HardwareSettings(BaseModel):
    device: Optional[DeviceIdentifier] = Field(
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
