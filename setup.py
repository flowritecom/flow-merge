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
        "accelerate==0.27.2",
        "huggingface-hub==0.21.1",
        "peft==0.9.0",
        "pydantic==2.4.2",
        "pyyaml==6.0.1",
        "safetensors==0.4.2",
        "torch==2.2.1",
        "transformers==4.38.1",
    ],
    package_data={
        "flow_merge": ["data/architectures/*"],
    },
    entry_points={"console_scripts": ["flow-merge=flow_merge.cli.manage:main"]},
)
