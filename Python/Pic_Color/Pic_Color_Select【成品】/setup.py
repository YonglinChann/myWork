# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='pic_color_lib',
    version='1.0.0',
    packages=find_packages(),
    description='一个用于图像处理和墨水屏数据生成的Python库',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Chen Yonglin',
    author_email='your.email@example.com',
    url='https://gitee.com/your_username/pic_color_lib', # 请替换为您的仓库URL
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-learn', # 现在在processor.py中使用了scikit-learn
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
)