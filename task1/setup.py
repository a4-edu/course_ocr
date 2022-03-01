import os
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


setup(
    name='course_ocr_t1',
    version='0.1.0',
    python_requires='>=3.7.0',
    packages=[''],
    description='Task1 for ABBYY OCR course',
    url='',
    author='Boris Zimka',
    author_email='boris.zimka@abbyy.com',
    install_requires=['numpy', 'matplotlib', 'shapely'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering'
    ],
)