# 3D images: 2D projections

**TODO**

Image segmentation with Anet is performed in 2D. 3D images are transformed into
2D images with a projection along Z. We provide a plugin to perform such projections,
where we provide different approaches.

The easiest method to achieve is the so-called **maximum intensity projection (MIP)**,
where for each XY position the highest pixel value along the z-axis is used.
We found that this approach can yield blurry cell boundaries because out-of-focus
contributions are considered. We proposed an alternative strategy, where projections
are performed by considered **focus measurements**, essentially how "sharp" the image
is. In this approach, we first blurry (out-of-focus) slices based on a global
focus measurement, and the resulting image is projected by using local
focus measurements to use regions wiht highest focus values.



## Focus projection
