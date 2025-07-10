from setuptools import setup, find_packages

setup(
    name='MatrixStriper',
    version='1.0.0',
    description='Pipeline de biclustering et compactage de matrice',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pandas',
    ],
    setup_requires=[
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
    },
    entry_points={
        'console_scripts': [
            # Pas de script console, mais support python -m MatrixStriper
        ],
    },
) 