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
        "pyyaml==6.0.1",
        "transformers==4.38.1",
        "torch==2.2.1",
        "safetensors==0.4.2",
        "peft==0.9.0",
        "accelerate==0.27.2",
        "tqdm==4.66.2",
        "pyright==1.1.347",
        "pytest==8.0.0",
        "huggingface-hub==0.21.1",
        "pydantic==2.4.2",
    ],
    package_data={
        "flow_merge": ["data/architectures/*"],
    },
    entry_points={"console_scripts": ["flow-merge=flow_merge.cli.manage:main"]},
)
