from setuptools import setup, find_packages
with open('README.md') as f:
    readme = f.read()
setup(
    name='butext',
    version='0.3.4',
    description='A collection of functions related to Lexical Analysis',
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'wordcloud'
    ]
)