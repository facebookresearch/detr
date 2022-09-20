#!/bin/bash
conda env create -f conda.yml
# shellcheck disable=SC1090
{
    source ~/miniconda-mio/etc/profile.d/conda.sh
}
conda activate cv-detr
pip install -e .
