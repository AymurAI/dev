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
        "gdown==4.4.0",
        "joblib==1.1.0",
        "textract==1.6.5",
        # "spacy==3.4.0",
        "spacy[transformers]==3.4.1",
        "spaczz==0.5.4",
        "datetime_matcher @ git+https://github.com/jedzill4/datetime_matcher",
        "scikit-learn==1.1.2",
        "jiwer==2.3.0",
        "datasets==2.4.0",
        "python-magic>=0.4.27",
        "redis==4.3.4",
        "unidecode==1.3.4",
        "sentencepiece==0.1.95",
        "python-multipart==0.0.5",
        "pydantic==1.8.2",
        "flair==0.11.3",
        "pytorch-lightning==1.8.3.post1",
        "torchtext==0.13.1",
        "tensorflow_hub==0.12.0",
        "tensorflow_text==2.11.0",
    ],
    extras_require={
        "": [
            "torch==1.12.1",
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
    classifiers=["Programming Language :: Python :: 3"],
)
