# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='pic_color_processor',
    version='0.1.0',
    packages=find_packages(),
    description='A library to process images for e-ink display data generation using dithering.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://gitee.com/your_username/your_repo', # Optional: Replace with your repo URL
    install_requires=[
        'numpy',
        'opencv-python',
        # scikit-learn is used in comments but not active code in processor.py
        # 'scikit-learn',
        # matplotlib is used in the original script but not in the core processor function
        # 'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)