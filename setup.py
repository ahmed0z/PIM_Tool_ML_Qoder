from setuptools import setup, find_packages

setup(
    name="autopatternchecker",
    version="1.0.0",
    description="Automated system that learns per-composite-key formats from a CSV database",
    author="AutoPatternChecker Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "hdbscan>=0.8.28",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.20.0",
        "sqlalchemy>=1.4.0",
        "python-dateutil>=2.8.0",
        "pytest>=7.0.0",
        "pyyaml>=6.0",
        "python-multipart>=0.0.6",
        "pydantic>=1.10.0",
    ],
    python_requires=">=3.10",
)