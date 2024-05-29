import os

class ApplicationConfig:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.device = os.getenv("DEVICE", "cpu")  # Default to 'cpu' if not set

    def set_hf_token(self, token: str):
        self.hf_token = token

    def set_device(self, device: str):
        self.device = device

config = ApplicationConfig()
