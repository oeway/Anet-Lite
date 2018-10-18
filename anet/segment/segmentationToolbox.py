"""
Created on Mon Oct  1 14:37:13 2018

@author: fmueller
"""

# IMPORTS
import os
import sys
from skimage.io import imsave
import shutil

# My Stuff
sys.path.append('/Volumes/PILON_HD2/fmueller/Documents/code/ImJoy_dev/segmentation/src')
import annotationImporter, maskGenerator


def create_folder(folder_new):
    if not os.path.isdir(folder_new):
        os.makedirs(folder_new)


def process_folder_fiji(path_open,channels,img_ext,annot_ext,masks_save):
    '''
    Function uses annotations generated in FIJI and creates mask based
    on the specified parameters. The resulting files are zipped and be
    used for training of a neural network with ImJoy
    '''

    ## Create folder to save results
    path_save_unet = os.path.join(path_open, 'unet_data_tmp')
    create_folder(path_save_unet)

    # Load data with FolderImporter
    folderImporter = annotationImporter.FolderImporter(channels=channels, data_category={'train':'train','test':'test','valid':'valid'},img_ext=img_ext,annot_ext =annot_ext)
    annotDict  = folderImporter.load(path_open)
    print('average roi size:', annotDict['roi_size'])

    # Generate binary masks
    binaryGen = maskGenerator.BinaryMaskGenerator(erose_size=5, obj_size_rem=500, save_indiv=False)

    # Loop over data categories and verify if corresponding folder exists
    for key_categ, categ in annotDict['config']['data_category'].items():

        # Folder to save category
        path_save_categ = os.path.join(path_save_unet, key_categ)
        create_folder(path_save_categ)

        # Loop over channels in data-categ
        for key_channel, channel in annotDict['config']['channels'].items():

           # The generate function uses as an input the sub-dictionary for one data-category and one channel
           maskDict = binaryGen.generate(annotDict[key_categ][key_channel])

           # Loop over masks and save specified ones
           for key_file, v in maskDict.items():
               for key_mask_sel in masks_save[key_channel]:

                   ## Get file name without extension
                   file_base, ext = os.path.splitext(key_file)

                   ## Create sample folder
                   sample_folder = os.path.join(path_save_categ, file_base)
                   create_folder(sample_folder)

                   ## Save label
                   img_save = maskDict[key_file][key_mask_sel]
                   file_name_save = os.path.join(sample_folder, '{}.png'.format(key_mask_sel))
                   imsave(file_name_save, img_save)

                   ## Save image
                   img_save = annotDict[key_categ][key_channel][key_file]['image']
                   file_name_save = os.path.join(sample_folder, '{}.png'.format(key_channel))
                   imsave(file_name_save, img_save)

    # Create zip file and delete original folder
    file_name_zip = os.path.join(path_open, 'unet_data')
    shutil.make_archive(file_name_zip, 'zip', path_save_unet)

    if os.path.isdir(path_save_unet):
        shutil.rmtree(path_save_unet)
