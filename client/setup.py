import setuptools

setuptools.setup(
    name="cbd_client",
    version="0.0.3",
    author="Michał Czuba",
    author_email="michal.czuba.1995@gmail.com",
    description="Package for remote connection with Cyber Bullying Detector",
    url="https://github.com/anty-filidor/CBD",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
    packages=["cbd_client"],
    python_requires=">=3.8",
    install_requires="requests==2.25.1",
)
