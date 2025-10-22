from setuptools import setup, find_packages

setup(
    name="samba",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core ML dependencies
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",

        # HuggingFace & datasets
        "transformers>=4.39.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",

        # mamba-ssm
        "mamba-ssm>=1.0.0",
        "causal-conv1d>=1.0.0",

        # Monitoring
        "wandb>=0.15.0",

        # Development
        "ipython",
        "jupyter",

        # GPU 필수 패키지
        "torch==2.5.1",
        "torchvision",
        "torchaudio",
        "triton>=2.0.0",
    ],
)
