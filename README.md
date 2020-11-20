# Time-varying weathering in Texture Space

This project is an implementation of [this paper](https://www.cs.tau.ac.il/~dcor/articles/2016/TW.pdf). 

Given an input image of a weathered texture, this project aims to synthesise a series of textures emulating a weathering and de-weathering processes, yielding a time-varying texture without any human interaction.

### Setup

- Clone the repository and `cd` into the project directory.
```bash
git clone https://github.com/ShivanshRakesh/time-varying-weathering-in-texture-space.git
cd project-image-imposters/
```
- Install the dependencies using:
```bash
pip install -r requirements.txt
```
- Launch Jupyter Notebook in the project root using `jupyter-notebook`.
- Navigate to `src/project.ipynb` and run all the cells to get the outputs for the default input textures.

### Usage

- Custom input textures can be processed by passing the image path(s) as a list to the `processImages()` function.
- Output images are written to `<project-root>/images/outputs/<image_name>/` if the `write_output` argument of `processImages()` is set `True`.

### Directory Structure
``` bash
.
├── documents
│   ├── DIP Presentation.pdf
│   └── DIP Presentation.pptx
├── guidelines.md
├── images
│   └── outputs
├── proposal.md
├── README.md
└── src
    ├── minimumCostPathFunc.py
    ├── project.ipynb
    └── textureTransfer.py
```

### Authors
- [Shivansh Rakesh](https://github.com/ShivanshRakesh)
- [Mohsin Mamoon Hafiz](https://github.com/MohsinMamoon)
- [Mohee Datta Gupta](https://github.com/MoheeDG23)
- [Srivathsan Baskaran](https://github.com/Srivathsan01)
