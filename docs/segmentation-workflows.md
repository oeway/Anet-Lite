#  Workflows
For each workflow, we provide an installation link for ImJoy to generate a dedicated
workspace with all necessary plugin, and the possibility to load a  pre-trained model.

This model can then be applied to test data to familiarize yourself with our approach.
The pre-trained model can also be used to initialize training on your own data.
This process (also called **transfer learning**) tends to reduce the time for training considerably,

For each workflow we provide a link to example data. These folders contain
* Data with the annotations (folders `train`, `valid`), and some test data (folder `test`).
  These folders can be processed with the `AnnotationGenerator` plugin.
* Processed data with the annotations being converted to images (folder `unet_data`).
  This folder can be used with the `Anet-Lite` plugin.

The folder structure for the segmentation of the cell membrane looks like This

```
.
├─ anet/                        # Folder for Anet plugin
│  ├─ train
│  │  ├─ img1/
│  │  │  ├─ cells.png
│  │  │  ├─ cells_mask_edge.png
│  │  ├─ ...
│  │
│  ├─ valid
│  │  ├─ img10/
│  │  │  ├─ cells.png
│  │  │  ├─ cells_mask_edge.png
│  │  ├─ ...
│  │
│  ├─ test1
│  │  ├─ C3-img21.tif
│  │  ├─ ...
│
├─ test/                            # Folder for AnnotationGenerator
│  ├─ C3-img20.tif
│  ├─ ...
│
├─ train/                           # Folder for AnnotationGenerator
│  ├─ C3-img1.tif
│  ├─ C3-img1__RoiSet.tif
│  ├─ ...
│
├─ valid/                           # Folder for AnnotationGenerator
│  ├─ C3-img10.tif
│  ├─ C3-img10__RoiSet.tif
│  ├─ ...
.
```


## Segmentation of cell membrane
This workflow allows to segment cellular membranes. The provided examples is for
cell cortex in the developing drosophila embryo.

**TODO**: SHOW EXAMPLE IMAGE

We provide some test data for a segmentation of the on [**Dropbox**](https://www.dropbox.com/sh/n4lq3f5xp1zmwtd/AABQmWyouXnTNnx2PL4Gz66Da?dl=0).
Data consists of a GFP stains of the membranes (e.g. `C3-img1.tif`) and the corresponding
annotations from FIJI (e.g. `C3-img1__RoiSet.zip`)

**TODO** should be provided as a Release. Should we pro

AnnotationGenerator:

![ImJoyScreenshot](/img/segment-param-cellcortex.png)

**TODO**  Link for ImJoy



## Segmentation of cells and nuclei
This workflow allows to segment cells and nuclei membranes. The provided examples
is for CellMask stain of cells, and a DAPI stain of nuclei.
cell cortex in the developing drosophila embryo.

**TODO**: SHOW EXAMPLE IMAGE

AnnotationGenerator:

![ImJoyScreenshot](/img/segment-param-cellsnuclei.png)

**TODO**  Link for ImJoy
