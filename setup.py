"""
Setup configuration for pyOPAC package.
"""
import setuptools
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "A pipeline for molecular property prediction and active learning"

# Read version from package
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), "opac", "__init__.py")
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setuptools.setup(
    name="pyOPAC",
    version=get_version(),
    author="Etinosa Osaro and Yamil Colon",
    author_email="eosaro@nd.edu, ycolon@nd.edu",
    description="A pipeline for molecular property prediction and active learning with SOAP descriptors",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/theOsaroJ/pyOPAC",
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "notebooks", "notebooks.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "ase>=3.21.0",
        "dscribe>=2.0.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "all": [
            "torchvision",
            "torch_geometric",
            "matplotlib",
            "tqdm",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyopac-modify-xyz=creating_the_xyz.modify:main",
            "pyopac-preprocess=opac.scripts.preprocess_data:main",
            "pyopac-compute-descriptors=opac.scripts.compute_descriptors:main",
            "pyopac-train=opac.scripts.train_model:main",
            "pyopac-predict=opac.scripts.predict_properties:main",
            "pyopac-generate=opac.scripts.generate_molecules:main",
            "pyopac-train-diffusion-generator=opac.scripts.train_diffusion_generator:main",
            "pyopac-active-learning=opac.scripts.run_active_learning:main",
        ],
    },
    keywords=[
        "molecular-property-prediction",
        "machine-learning",
        "active-learning",
        "soap-descriptors",
        "computational-chemistry",
        "molecular-descriptors",
        "neural-networks",
    ],
    project_urls={
        "Bug Reports": "https://github.com/theOsaroJ/pyOPAC/issues",
        "Source": "https://github.com/theOsaroJ/pyOPAC",
        "Documentation": "https://github.com/theOsaroJ/pyOPAC#readme",
    },
)
