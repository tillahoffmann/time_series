from setuptools import setup, find_packages

setup(
    name='time_series',
    packages=find_packages(),
    install_requires=[
        "bayespy>=0.5",
        "matplotlib>=2.0",
        "numpy>=1.11",
        "scikit-learn>0.19",
    ]
)
