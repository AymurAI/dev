from setuptools import setup, find_packages

setup(
    name="aymurai",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="collective.ai & datagenero.org",
    author_email="aymurai@datagenero.org",
    install_requires=[
        "more-itertools>=8.12.0",
        "odfpy>=1.4.1",
        "gdown>=4.6.0",
        "joblib>=1.1.0",
        "textract>=1.6.5",
        "spacy[transformers]==3.4.1",
        "spaczz>=0.5.4",
        "datetime_matcher @ git+https://github.com/jedzill4/datetime_matcher",
        "scikit-learn>=1.1.2",
        "jiwer>=2.3.0",
        "datasets>=2.4.0",
        "python-magic>=0.4.27",
        "redis>=4.3.4",
        "unidecode>=1.3.4",
        "sentencepiece>=0.1.95",
        "flair @ git+https://github.com/flairNLP/flair",
        "pytorch-lightning==1.8.3.post1",
        "torchtext==0.13.1",
        "tensorflow_hub>=0.12.0",
        "tensorflow_text>=2.11.0",
        "fastapi>=0.97.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.5",
        "pydantic>=1.8.2",
        "httpx>=0.24.1",
        "faker==18.11.2",
        "xmltodict==0.13.0",
        "cachetools==5.3.2",
        "diskcache==5.6.3",
        "mammoth==1.6.0",
    ],
    extras_require={
        "": [
            "torch==1.12.1",
            "rich==12.6.0",
        ],
        "gpu": [
            "torch @ https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp310-cp310-linux_x86_64.whl",
            "spacy[cuda113]==3.4.1",
        ],
        "dev": [
            # "rich[jupyter]==12.6.0",
            "tensorboard==2.11.0",
            "jupyterlab==3.4.3",
            "ipywidgets==8.0.1",
            "matplotlib==3.5.2",
            "seaborn==0.11.2",
            "plotly==5.9.0",
        ],
    },
    package_data={"": ["*.yml", "*.yaml", "*.csv"]},
    include_package_data=True,
    scripts=["aymurai/api/main.py"],
    classifiers=["Programming Language :: Python :: 3"],
)
