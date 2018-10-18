'''
Introduction
============

Module to produce segmentation masks. These modules use as an input annotation
dictionaries created by the separate module 'AnnotationImporter'.

Different types of masks can be generated
(1) binary masks for the surface and the edge
(2) Weighted edge masks

'''
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from skimage import morphology
from skimage import draw as drawSK
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# Info about the module
__version__ = '0.1.2'
__author__ = 'Florian MUELLER, Aubin SAMACOITS'
__email__ = 'muellerf.research@gmail.com'

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class MaskGenerator():
    '''Base class for mask generators.'''

    def __init__(self):
        pass

    def generate(self, annotDic):
        '''
            Generate the mask, return a dictonary.
        '''
        raise NotImplementedError('No load function defined for this class!')

    def plot(self):
        pass

    def save(self, path, suffix=''):
        ''' Save mask to images'''
        pass

class BinaryMaskGenerator(MaskGenerator):
    ''' Create binary masks from an annotation dictionary. Uses as and input
    directory containing images ('image') and the regions ('roi'). Creates as
    an output a dictionart. Which masks are created depends on the annotation
    type. If masks are

    ... polygons, then an edge mask and a filled mask are created.
    ... freelines, then only an edge mask is created
    '''

    def __init__(self, erose_size=5, obj_size_rem=500, save_indiv=False):
        self.erose_size = erose_size
        self.obj_size_rem = obj_size_rem
        self.save_indiv = save_indiv

    def generate(self, annotDic):
        '''
        Input is a dictonary containing files with two fields 'image' and 'roi'.
        This dictonary is usually obtained as sub-dictonary from the anntoation
        dictonary created by the AnnotationImporter.

        Returns a dictonary (maskDict) with three entries
        'mask_fill' with the filled mask for the annotated objects,
        'mask_edge' with the edge of the annotated objects, and (if desired)
        'mask_edge_indiv' containing 3D arrays where the edge of each
        annotated object is saved sepatately. The latter is needed to calculate the
        weighted edge mask.
        '''
        maskDict = {}

        for annot_key, annot_data in annotDic.items():

            # ROI dictionary
            roi_dict = annot_data['roi']

            # Get dimensions of image and created masks of same size

            image_size = annot_data['image'].shape
            print(image_size)

            # Filled masks and edge mask for polygons
            mask_fill = np.zeros(image_size, dtype=np.uint8)
            mask_edge = np.zeros(image_size, dtype=np.uint8)

            rr_all = []
            cc_all = []

            if self.save_indiv is True:
                mask_edge_indiv = np.zeros(
                    (image_size[0], image_size[1], len(roi_dict)), dtype=np.bool)
                mask_fill_indiv = np.zeros(
                    (image_size[0], image_size[1], len(roi_dict)), dtype=np.bool)

            # Image used to draw lines - for edge mask for freelines
            im_freeline = Image.new('1', image_size, color=0)
            draw = ImageDraw.Draw(im_freeline)

            # Loop over all roi
            i_roi = 0
            for roi_key, roi in roi_dict.items():

                print(roi_key)

                roi_pos = roi['pos']

                # Check region type

                # freeline - line
                if roi['type'] == 'freeline':

                    # Loop over all pairs of points to draw the line

                    for ind in range(roi_pos.shape[0] - 1):
                        line_pos = ((roi_pos[ind, 1], roi_pos[ind, 0], roi_pos[
                            ind + 1, 1], roi_pos[ind + 1, 0]))
                        draw.line(line_pos, fill=1, width=self.erose_size)

                # freehand - polygon
                elif roi['type'] == 'freehand' or roi['type'] == 'polygon' or roi['type'] == 'polyline':

                    # Draw polygon
                    rr, cc = drawSK.polygon(roi_pos[:, 0], roi_pos[:, 1])

                    # Make sure it's not outside
                    rr[rr < 0] = 0
                    rr[rr > image_size[0] - 1] = image_size[0] - 1

                    cc[cc < 0] = 0
                    cc[cc > image_size[0] - 1] = image_size[0] - 1

                    # Test if this region has already been added
                    if any(np.array_equal(rr, rr_test) for rr_test in rr_all) and any(np.array_equal(cc, cc_test) for cc_test in cc_all):
                        # print('Region #{} has already been used'.format(i +
                        # 1))
                        continue

                    rr_all.append(rr)
                    cc_all.append(cc)

                    # Generate mask
                    mask_fill_roi = np.zeros(image_size, dtype=np.uint8)
                    mask_fill_roi[rr, cc] = 1

                    # Erode to get cell edge - both arrays are boolean to be used as
                    # index arrays later
                    mask_fill_roi_erode = morphology.binary_erosion(
                        mask_fill_roi, np.ones((self.erose_size, self.erose_size)))
                    mask_edge_roi = (mask_fill_roi.astype('int') -
                                     mask_fill_roi_erode.astype('int')).astype('bool')

                    # Save array for mask and edge
                    mask_fill[mask_fill_roi_erode] = 1
                    mask_edge[mask_edge_roi] = 1

                    if self.save_indiv is True:
                        mask_edge_indiv[:, :, i_roi] = mask_edge_roi.astype('bool')
                        mask_fill_indiv[
                            :, :, i_roi] = mask_fill_roi_erode.astype('bool')
                        i_roi = i_roi + 1

                else:
                    roi_type = roi['type']
                    raise NotImplementedError(f'Mask for roi type "{roi_type}" can not be created')

            del draw

            # Convert mask from free-lines to numpy array
            mask_edge_freeline = np.asarray(im_freeline)
            mask_edge_freeline = mask_edge_freeline.astype('bool')

            # Post-processing of fill and edge mask - if defined
            if np.any(mask_fill):

                # (1) remove edges , (2) remove small  objects
                mask_fill = mask_fill & ~mask_edge
                mask_fill = morphology.remove_small_objects(
                    mask_fill.astype('bool'), self.obj_size_rem)

                # For edge - consider also freeline edge mask
                mask_edge = mask_edge.astype('bool')
                mask_edge = np.logical_or(mask_edge, mask_edge_freeline)

                # Assign to dictionary for return
                maskDict[annot_key] = {'mask_edge': mask_edge, 'mask_fill': mask_fill.astype(
                    'bool')}

                if self.save_indiv is True:
                    maskDict[annot_key]['mask_edge_indiv'] = mask_edge_indiv
                    maskDict[annot_key]['mask_fill_indiv'] = mask_fill_indiv
                else:
                    maskDict[annot_key]['mask_edge_indiv'] = np.zeros(image_size+(1,), dtype=np.uint8)
                    maskDict[annot_key]['mask_fill_indiv'] = np.zeros(image_size+(1,), dtype=np.uint8)

            # Only edge mask present
            elif np.any(mask_edge_freeline):
                maskDict[annot_key] = {'mask_edge': mask_edge_freeline,'mask_fill': mask_fill.astype(
                    'bool')}

                maskDict[annot_key]['mask_edge_indiv'] = np.zeros(image_size+(1,), dtype=np.uint8)
                maskDict[annot_key]['mask_fill_indiv'] = np.zeros(image_size+(1,), dtype=np.uint8)

            else:
                raise NotImplementedError('No mask has been created.')

        return maskDict


class WeightedEdgeMaskGenerator(MaskGenerator):
    ''' Create a weighted edge mask. Uses as and input a dictonary that contains
    for each image, ad 3D array, where the edge mask of each individual object
    is stored. This is generated with the BinaryMaskGenerator. The output is a
    dictonary contained the weighted edge mask for each image.
    '''

    def __init__(self, sigma=8, w0=10):
        self.sigma = sigma
        self.w0 = w0

    def generate(self, annotDic):
        '''
        Creates weights for edges that depend on distance to two closests cells.
        Reference: https://arxiv.org/abs/1505.04597
        Requires that binary weights are calculated first and the edge of each
        cell saved with the option flag_save_indiv=True
        Results are saved in a dictonary with the key 'mask_edge_weighted'
        '''

        from scipy import ndimage

        maskDict = {}

        for key, annot in annotDic.items():

            mask_fill = annot['mask_fill']
            mask_edge_indiv = annot['mask_edge_indiv']

            # Calculating the weigth w that balance the pixel frequency
            x = (mask_fill > 0).astype('int')

            # Percentage of image being a cell
            ratio = float(x.sum()) / float(x.size - x.sum())

            if ratio < 1.0:
                wc = (1 / ratio, 1)
            else:
                wc = (1, 1 / ratio)

            # Calculate the distance map from each pixel to every cell
            dist_mat = np.ones(np.shape(mask_edge_indiv))
            image_ones = np.ones(np.shape(mask_fill))

            for i_cell in range(mask_edge_indiv.shape[-1]):

                edge_cell_inverted = image_ones - \
                    1 * mask_edge_indiv[:, :, i_cell]
                dist_mat[:, :, i_cell] = ndimage.distance_transform_edt(
                    edge_cell_inverted)

            # Sort distance map and use only the two closest cells and add them
            # up
            dist_map = np.sum(np.sort(dist_mat)[:, :, (0, 1)], 2)

            # Calculated exponential weight for each pixel
            exp_weigth = self.w0 * np.exp(-(dist_map)**2 / (2 * self.sigma**2))

            # Calculate frequency weight
            wc_map = mask_fill * wc[0] + (1 - mask_fill) * wc[1]
            mask_edge = wc_map + exp_weigth

            # Sum of both weights
            # Note: saved as float 16 - in order to plot has to be converted to
            # float32
            maskDict[key] = {'mask_edge_weighted': mask_edge.astype('float16')}

        return maskDict


class DistanceMapGenerator(MaskGenerator):
    ''' Create a distance transform from the edge.
    '''

    def __init__(self, truncate_distance=None):
        self.truncate_distance = truncate_distance

    def generate(self, annotDic):
        '''
        Creates a distance map with truncated distance to the edge of the cell.
        '''

        from scipy import ndimage

        maskDict = {}

        for key, annot in annotDic.items():

            mask_fill_indiv = annot['mask_fill_indiv']
            mask_edge_indiv = annot['mask_edge_indiv']
            dist_mat = np.ones(np.shape(mask_fill_indiv))

            for i_cell in range(mask_fill_indiv.shape[-1]):
                img_cell = mask_edge_indiv[
                    :, :, i_cell] + mask_fill_indiv[:, :, i_cell]

                dist_cell = ndimage.distance_transform_edt(img_cell)
                if self.truncate_distance:
                    dist_cell[dist_cell >
                              self.truncate_distance] = self.truncate_distance
                dist_mat[:, :, i_cell] = dist_cell

            dist_map = np.sum(dist_mat, 2)

            # Note: saved as float 16 - in order to plot has to be converted to
            # float32
            maskDict[key] = {'distance_map': dist_map.astype('float16')}

        return maskDict


if __name__ == '__main__':
    from annotationImporter import ZipFileImporter, save_dict_to_hdf5, load_dict_from_hdf5
    # Read annotation results from a .zip archive that were generated with
    # Fiji.
    zipImporter = ZipFileImporter(
        channels={'cells': 'Cy5', 'nuclei': 'DAPI'}, data_category={'train': 'Training', 'valid': 'Validation'})

    print('loading from zip file ../testdata/Data_FileImporter.zip')
    annotDict = zipImporter.load('../testdata/Data_FileImporter.zip')
    print(annotDict.keys())

    binaryGen = BinaryMaskGenerator(
        erose_size=5, obj_size_rem=500, save_indiv=True)

    # The generate function uses as an input the sub-dictonary for one
    # data-category and one channel
    maskDict = binaryGen.generate(annotDict['train']['cells'])

    for k, v in annotDict['train']['cells'].items():
        # The update function can be used to add the mask dictonary to the
        # loaded annotation dictonary
        v.update(maskDict[k])

    # truncated edge
    distanceGen = DistanceMapGenerator(truncate_distance=20)
    maskDict = distanceGen.generate(annotDict['train']['cells'])

    for k, v in annotDict['train']['cells'].items():
        # The update function can be used to add the mask dictionary to the
        # loaded annotation dictonary
        v.update(maskDict[k])

    # weighted edge
    weightedGen = WeightedEdgeMaskGenerator(sigma=8, w0=10)
    maskDict = weightedGen.generate(annotDict['train']['cells'])

    for k, v in annotDict['train']['cells'].items():
        # The update function can be used to add the mask dictonary to the
        # loaded annotation dictonary
        v.update(maskDict[k])

    print(annotDict['train']['cells'].keys())
