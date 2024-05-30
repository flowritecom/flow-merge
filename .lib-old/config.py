import os


class ApplicationConfig:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.device = "TBD"
        ## FIXME: bring device setting here and read from here at the implementation level, skip passing
        ## Beware some places don't mention device but have "cpu", FIND THEM
        ## We also have mixed DeviceIdentifier Literal[str] = 'cpu', 'cuda'
        ## Hugging Face's DeviceSomething - competing type ?

    def set_hf_token(self, token: str):
        self.hf_token = token


config = ApplicationConfig()
