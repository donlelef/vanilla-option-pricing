from setuptools import setup, find_packages

from docs.conf import version

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vanilla_option_pricing',
    version=version,
    packages=find_packages(),
    url='https://vanilla-option-pricing.readthedocs.io/en/latest/',
    license='MIT',
    author='Emanuele Fabbiani',
    author_email='emanuele.fabbiani@xtreamers.io',
    description='Stochastic models to price financial options',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['pandas', 'py_vollib', 'scipy', 'numpy', 'py_lets_be_rational'],
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
