"""Setup script for llm-dataprep package"""

from setuptools import setup, find_packages
from pathlib import Path

# Lese README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-dataprep",
    version="0.1.0",
    author="DataPrep Team",
    description="Modern LLM Pre-Training Data Preparation Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-dataprep",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "nltk>=3.8",
        "langdetect>=1.0.9",
        "xxhash>=3.2.0",
        "datasketch>=1.5.9",
        "pyarrow>=12.0.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "pdf": [
            "pypdf2>=3.0.0",
            "pdfplumber>=0.9.0",
        ],
        "latex": [
            "pylatexenc>=2.10",
        ],
        "video": [
            "av>=10.0.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "quality": [
            "detoxify>=0.5.0",
        ],
        "all": [
            "pypdf2>=3.0.0",
            "pdfplumber>=0.9.0",
            "pylatexenc>=2.10",
            "av>=10.0.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "detoxify>=0.5.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm pre-training data-preprocessing phi-3 multimodal",
)
