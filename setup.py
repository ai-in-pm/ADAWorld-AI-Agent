from setuptools import setup, find_packages

setup(
    name="adaworld",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'agno>=0.1.0',
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'gymnasium>=0.29.0',
        'pygame>=2.5.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'pytest>=7.0.0',
    ],
    author="ADAWorld Team",
    author_email="adaworld@example.com",
    description="Adaptable World Models with Latent Actions using Agno framework",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaworld",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'adaworld-train=adaworld.agent.trainer:main',
            'adaworld-viz=adaworld.utils.visualizer:main',
        ],
    },
)