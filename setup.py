from setuptools import setup, find_packages

setup(
    name='komisan',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "optbinning==0.18.0",
        "imblearn",
        "scikit-learn==1.1.3",
        "statsmodels"
    ],
)
