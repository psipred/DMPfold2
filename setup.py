import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dmpfold",
    version="0.1",
    author="UCL Bioinformatics Group",
    author_email="psipred@cs.ucl.ac.uk",
    description="Protein structure prediction with the DMPfold2 method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psipred/DMPfold2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="protein structure prediction deep learning evolutionary constraint",
    scripts=["bin/dmpfold"],
    install_requires=["numpy", "scipy", "PeptideBuilder"],
)
