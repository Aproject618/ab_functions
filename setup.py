from setuptools import setup, find_packages

setup(
    name='bootstrap_ab_test',
    version='0.1.0',
    description='Bootstrap-based A/B test analyzer',
    author='Ramis Sungatullin',
    author_email='aproject618@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'plotly',
        'tqdm'
    ],
    python_requires='>=3.7',
)
