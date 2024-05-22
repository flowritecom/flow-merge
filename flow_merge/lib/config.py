import os


class CentralConfig:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")

    def set_hf_token(self, token: str):
        self.hf_token = token


config = CentralConfig()
