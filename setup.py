import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microgradnp",
    version="0.1.0",
    author="Daniel Cox",
    author_email="danielpcox@gmail.com",
    description="A small autodiff and neural network library patterned on Karpathy's micrograd,"
                "but with support for tensors expressed as numpy arrays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel/microgradnp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
)
