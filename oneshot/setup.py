from setuptools import setup, find_packages

setup(
    name="oneshot",
    version="0.1.0",
    description="Reinforcement Learning for Franka Robot",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "open3d",
        "scikit-learn",
        "numpy-quaternion",
    ],
    package_data={
        "oneshot": [
        ]
    },
    entry_points={
        "console_scripts": [
            "franka-pnp=oneshot.standalone.franka_pnp:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
