#!/bin/bash
# Installs scipy, matplotlib, the scorers and spaCy into the user site, inside the container:
#   apptainer exec liveseed.sif bash install_extras.sh        (set PYTHONUSERBASE to choose the user site)
set -e
cd "$(dirname "$0")"
python -m pip install --quiet --user --no-deps -r <(grep -v '^#' requirements-extra.txt | sed 's/ *#.*//' | grep .)
python -m pip install --quiet --user 'scipy==1.16.3' 'matplotlib==3.10.9' 'datasets==2.21.0' 'pyarrow==17.0.0' 'dill<0.4' spacy
python -m spacy download en_core_web_sm
# the hpsv2 wheel on PyPI ships without its BPE vocabulary: copy it from open_clip
python -c 'import os, shutil, open_clip, hpsv2; d = os.path.join(os.path.dirname(hpsv2.__file__), "src/open_clip/bpe_simple_vocab_16e6.txt.gz"); os.path.exists(d) or shutil.copy(os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz"), d)'
python -c 'import scipy, matplotlib, pyarrow, hpsv2, hpsv2.img_score, ImageReward, clip, open_clip, datasets, spacy; from ImageReward.models.BLIP import blip_pretrain; print("scoring packages ok")'
