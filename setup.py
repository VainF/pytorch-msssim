import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_msssim",
    version="1.0.0",
    author="Gongfan Fang",
    author_email="gongfan@u.nus.edu",
    description="Fast and differentiable MS-SSIM and SSIM for pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VainF/pytorch-msssim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch']
)
