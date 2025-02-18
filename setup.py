import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyOPAC",
    version="0.1.0",
    author="Etinosa Osaro and Yamil Colon",
    author_email="eosaro@nd.edu, ycolon@nd.edu",
    description="A pipeline for molecular property prediction and active learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theOsaroJ/pyOPAC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "torch",
        "scikit-learn",
        "ase",
        # Note: RDKit and Open Babel are not available via PyPI and must be installed separately. see install.sh for conda help
    ],
    entry_points={
        "console_scripts": [
            # Optionally create CLI commands:
            "modify_xyz=creating_the_xyz.modify:main",
            "train_model=opac.scripts.train_model:main",
            "preprocess_data=opac.scripts.preprocess_data:main",
            "compute_descriptors=opac.scripts.compute_descriptors:main",
            "predict_properties=opac.scripts.predict_properties:main",
        ],
    },
)
