from setuptools import setup

setup(
    name="icl",
    version="1.0.0",
    description=(
        "Code for 'How does the pretraining distribution shape in-context learning?'"
    ),
    license="Apache-2.0",
    packages=["icl"],
    package_data={"icl": ["configs/*.yaml"]},
    python_requires=">=3.10",
)
