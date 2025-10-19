from setuptools import setup, find_packages
setup(
    name='butext',
    version='0.3.4',
    description='A collection of functions related to Lexical Analysis',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'wordcloud'
    ]
)