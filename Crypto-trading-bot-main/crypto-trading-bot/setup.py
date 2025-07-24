"""
Setup configuration for the Cryptocurrency Trading Bot
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced Cryptocurrency Trading Bot with AI Integration"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "pandas>=1.5.0",
            "numpy>=1.24.0", 
            "ccxt>=4.0.0",
            "aiohttp>=3.8.0",
            "sqlalchemy>=2.0.0",
            "pyyaml>=6.0"
        ]

# Get version
def get_version():
    version_file = os.path.join("config", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="crypto-trading-bot",
    version=get_version(),
    author="Trading Bot Team",
    author_email="team@tradingbot.com",
    description="Advanced Cryptocurrency Trading Bot with AI Integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-trading-bot",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/crypto-trading-bot/issues",
        "Documentation": "https://crypto-trading-bot.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/crypto-trading-bot",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Framework :: AsyncIO",
    ],
    keywords="cryptocurrency trading bot bitcoin ethereum ai machine-learning algorithmic-trading",
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "bandit>=1.7.5",
            "safety>=2.3.0",
        ],
        "dashboard": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "matplotlib>=3.7.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "xgboost>=1.7.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.17.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "jaeger-client>=4.8.0",
            "opentelemetry-api>=1.19.0",
        ],
        "all": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "scikit-learn>=1.3.0",
            "boto3>=1.28.0",
            "prometheus-client>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-bot=run_bot:main",
            "trading-bot-test=test_strategies:main",
            "trading-bot-config=config.settings:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.json"],
        "strategies": ["*.py"],
        "utils": ["*.py"],
        "database": ["*.py"],
        "ai_analysis": ["*.py"],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    # Additional metadata
    maintainer="Trading Bot Team",
    maintainer_email="team@tradingbot.com",
    download_url="https://github.com/yourusername/crypto-trading-bot/archive/v1.0.0.tar.gz",
)