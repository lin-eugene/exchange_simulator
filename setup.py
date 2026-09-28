from setuptools import setup, find_packages
setup(
    name="exchange_simulator",
    version="0.1",
    packages=find_packages('exchange_simulator', 'exchange_simulator.*'),
    package_dir={'': 'exchange_simulator'},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
        "streamlit"
    ]
)
