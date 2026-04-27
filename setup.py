"""
Setup configuration for cifar10-cnn-classifier package.
Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cifar10-cnn-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade CIFAR-10 image classification with ResNet-inspired CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cifar10-cnn-classifier",
    packages=find_packages(exclude=["notebooks", "results", "checkpoints"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "cifar10-train=main:main",
        ],
    },
)
