# Time-varying weathering in Texture Space

This project is an implementation of [this paper](https://www.cs.tau.ac.il/~dcor/articles/2016/TW.pdf). 

Given an input image of a weathered texture, this project aims to synthesise a series of textures emulating a weathering and de-weathering processes, yielding a time-varying texture without any human interaction.

### Objective

![](/images/example.png)

Given an input image of a weathered texture, this project aims to synthesize a series of textures emulating weathering and de-weathering processes, yielding a time-varying texture (like in the image above). This is done by computing an estimated age map of the texture based on the prevalence of similar patches in the texture. Further, using this age map, an intact texture is generated to achieve the desired results. 

To produce de-weathered textures, the age map is manipulated to control an interpolation of the intact texture and the input texture. Weathered textures are synthesized by extrapolating the differences between the input texture and the intact texture.

This project aims at achieving these goals without any user interaction or assistance.

### Setup

- Clone the repository and `cd` into the project directory.
```bash
git clone https://github.com/ShivanshRakesh/time-varying-weathering-in-texture-space.git
cd time-varying-weathering-in-texture-space/
```
- Install the dependencies using:
```bash
pip install -r requirements.txt
```
- Launch Jupyter Notebook in the project root using `jupyter-notebook`.
- Navigate to `src/project.ipynb` and run all the cells to get the outputs for the default input textures.

### Usage

- Custom input textures can be processed by passing the image path(s) as a list to the `run()` function.
- Output images are written to `<project-root>/images/outputs/<image_name>/` if the `write_output` argument of `run()` is set `True`.

### Directory Structure
``` bash
.
├── images
│   └── outputs
├── README.md
└── src
    ├── minimumCostPathFunc.py
    ├── project.ipynb
    ├── project_withTests.ipynb
    └── textureTransfer.py
```

### Authors
- [Shivansh Rakesh](https://github.com/ShivanshRakesh)
- [Mohsin Mamoon Hafiz](https://github.com/MohsinMamoon)
- [Mohee Datta Gupta](https://github.com/MoheeDG23)
- [Srivathsan Baskaran](https://github.com/Srivathsan01)
