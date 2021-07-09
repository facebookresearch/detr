import os
from codecs import open as copen
from setuptools import setup


# Get the long description from README.md
with copen(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), mode='r', encoding='utf-8') as f:
    long_description = f.read()


# Version
__version__ = 'v0.2'


setup(
    name='detr',
    version=__version__,
    description="DETR: End-to-End Object Detection with Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/detr",
    author="Facebook Research",
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    # Add here the package's dependencies
    install_requires=[
        'cython',
        'pycocotools @ git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools',
        'submitit',
        'torch>=1.5.0',
        'torchvision>=0.6.0',
        'panopticapi @ git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi',
        'scipy',
        'onnx',
        'onnxruntime',
    ],
    test_deps=[],
)
