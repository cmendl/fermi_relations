from setuptools import setup

setup(
    name="fermi_relations",
    version="0.0.1",
    author="Christian B. Mendl",
    author_email="christian.b.mendl@gmail.com",
    packages=["fermi_relations"],
    url="https://github.com/cmendl/fermi_relations",
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
