from setuptools import find_packages, setup

setup(
    name='pycap',
    version='0.0.1',
    namespace_packages=['pycap'],
    packages=find_packages('.'),
    install_requires=[
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "scipy",
        "sklearn",
        "tensorflow",
        "keras",
        "pillow",
        "matplotlib"
    ],
    package_data={
        # If any package contains *.json, include them:
        '': ['*.json', '*.html']
    }
)
