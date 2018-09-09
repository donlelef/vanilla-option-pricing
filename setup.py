from setuptools import setup, find_packages

from docs.conf import version

setup(
    name='vanilla_option_pricing',
    version=version,
    packages=find_packages(),
    url='',
    license='MIT',
    author='Emanuele Fabbiani',
    author_email='emanuele.fabbiani@xtreamers.io',
    description='A very simple library to calibrate mean-reverting stochastic models and compute their variance. '
                'Includes vanilla option pricing functionalities.',
    install_requires=['pandas', 'py_vollib', 'scipy', 'numpy', 'py_lets_be_rational'],
    test_suite='nose.collector',
    tests_require=['nose']
)
