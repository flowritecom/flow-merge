from pydantic import BaseModel, Field


class HfHubSettings(BaseModel):
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading models from the Hugging Face Hub or not.",
    )

    def _unpack(self):
        return self.trust_remote_code
