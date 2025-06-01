from setuptools import setup, find_packages

setup(
    name='topic_classifier',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'transformers',
        'sentence-transformers',
        'scikit-learn',
        'nltk',
        'torch',
    ],
    description='Topic classification using zero-shot and semantic similarity',
    author='A. Janada',
    url='https://github.com/yourusername/topic-classifier',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
