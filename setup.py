from setuptools import find_packages, setup

setup(
    name="flow-merge",
    version="0.1.0",
    description="Module for model merging",
    author="Flow AI",
    author_email="nothere@flowrite.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "accelerate==0.32.1",
        "huggingface-hub==0.23.4",
        "peft==0.11.1",
        "pydantic==2.8.2",
        "pyyaml==6.0.1",
        "safetensors==0.4.3",
        "torch==2.3.1",
        "transformers==4.42.4",
    ],
    package_data={
        "flow_merge": ["data/architectures/*"],
    },
    entry_points={"console_scripts": ["flow-merge=flow_merge.cli.manage:main"]},
)
