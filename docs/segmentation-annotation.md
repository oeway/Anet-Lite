# Image Annotation

In this step, the structures of interest, e.g. cell and nuclei, are outline in
the images to generate data that will be used to train the neural network.

For the moment, we support annotations from [FIJI](https://fiji.sc/).
Using annotations from other software packages is in principle possible but
will require the implementation of dedicated wrappers.

We provide some test data for a segmentation of the cell cortex in the developing
drosophila embryo on [**Dropbox**](https://www.dropbox.com/sh/n4lq3f5xp1zmwtd/AABQmWyouXnTNnx2PL4Gz66Da?dl=0).

## File organization

Data has to be split into two folders **train** and **test**. The train folder will
be used to train the neural network, the test folder to continuously monitor how
well the training worked. For more details see our [deep learning primer](deeplearning.md).

There is no simple rule for how many images / annotated cells or nuclei you will need
to obtain good results. For standard segmentation of adherent cells, we obtained
good results with a training set of 5 images (with up to 10-15 cells per image), and test set of 2 images.

## Annotation with FIJI

We create the annotations with FIJI.

1.  Open FIJI
2.  Open the **ROI manager**: `Analyze` > `Tools` > `ROI manager`
3.  Open image that you will annotate.
4.  Select the annotation tool of choice, e.g. freehand or polygon.
5.  Outline first structure of interest. When done, press `Add(t)` in ROI manager to
    add outline.
6.  Proceed with all other structures. Enabling "Show all", will show all defined regions.
7.  Save regions by highlighting all regions in the list and pressing on More >> Save ...
8.  If only one region is save, this will created a file with the extension .roi, if
    multiple regions will be save this will create a .zip file. As a file-name choose
    the name of the annotated image, followed by a suffix such as `**_ROI.zip**`. If you have
    different structures (e.g. nuclei and cells), choose the suffix accordingly, e.g.
    `**_cells_ROI.zip**` and `**_nuclei_ROI.zip**`.

**IMPORTANT**: all structures, e.g. nuclei, have to be annotated. Unwanted elements,
e.g. nuclei touching the cell border, can be removed in a post-processing step.

## Convert annotations to images

Once you have annotated the images, you have to convert these annotations to images which
can be used as an input for the A-net. We provide a dedicated ImJoy
plugin to perform this task. This plugin has different **tags**, which render the plugin
 for a given segmentation tasks. You only have to specify a few key properties of your data.

The screenshot shown below shows the plugin interface for the segmentation of the
cell membrane in the example data. Note that here we only have one channel.

![Alt text](/img/seg-param-cellcortex.png)

For the example above, you have to specify

1. Unique identifier for each channel (here there is only one channel with the identifier `C3-`)
2. File extension of your images (`.tif` in this case)
3. File extension of the annotations (FIJI annotations with `_RoiSet.zip`)

Once you specified these parameters you can convert an entire folder with annotations
by pressing on the plugin itself (the blue text `AnnotationImporter`). Two possibilities exist

1. You don't select any window in the main interface of ImJoy. Then you will be asked to specify a folder.
0. You specify first a folder that you would like to process. Press on the + symbol in the
upper left corner, select `Load files through the python plugin engine`. This will create a window on your ImJoy workspace
corresponding to the folder that you want to process. You can now select this window (the title bar wil turn blue), and press
on the plugin name. The plugin will then process this folder.
0. The plugin will open all annotation files, create the necessary mask images, and
save them together with the corresponding images in a new folder `unet_data` in the processed folder.
This directory can be used as an input directory for the training
