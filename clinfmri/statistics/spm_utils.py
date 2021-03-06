#! /usr/bin/env python
##########################################################################
# CAPS - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System modules
import os
import nibabel
import scipy.io
import matplotlib
import numpy

# Set matplotlib backend
matplotlib.use("AGG")

import matplotlib.pyplot as plt


def spm_save_design(spm_mat_file, output_directory, no_int=0):
    """ Create a snap with the design matrix values.

    <unit>
        <input name="spm_mat_file" type="File" desc="The spm mat file
            containing the desing matrix." />
        <input name="output_directory" type="Directory" desc="The destination
            folder." />
        <output name="spm_design_snap" type="File" desc="A snapshot of the
            design matrix." />
    </unit>
    """
    # Get the design matrix
    spmmat = scipy.io.loadmat(spm_mat_file, struct_as_record=False)
    designmatrix = spmmat["SPM"][0][0].xX[0][0].X
    regressor_names = spmmat["SPM"][0][0].xX[0][0].name

    # Create a snapshot of the design matrix
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.imshow(designmatrix, cmap="flag", aspect="auto", interpolation="none")
    plt.xlabel("parameters")
    plt.ylabel("images")
    plt.title("Statistical analysis: Design")

    # add legend
    ax.set_xticks(numpy.arange(designmatrix.shape[1]) + 0.5)
    xlabels = [""] * designmatrix.shape[1]
    if no_int > designmatrix.shape[1]:
        no_int = 0
    for index, label in enumerate(regressor_names[0][:-no_int]):
        xlabels[index] = label[0].replace("Sn(1) ", "").replace("*bf(1)", "")
    ax.set_xticklabels(xlabels, rotation=60, ha="right")

    # save
    spm_design_snap = os.path.join(output_directory, "spm_design.png")
    plt.savefig(spm_design_snap, bbox_inches='tight')

    return spm_design_snap


def spm_image_nii_encoding(con_images, spmT_images, ess_images, spmF_images,
                           contrasts, output_directory):
    """ Encode the output spm images in compress nifti format and rename files
    to include the contrast name.

    <unit>
        <input name="con_images" type="List" content="File"
            desc="Contrast images from a t-contrast." />
        <input name="spmT_images" type="List" content="File"
            desc="Stat images from a t-contrast." />
        <input name="ess_images" type="List" content="File"
            desc="Contrast images from an F-contrast." />
        <input name="spmF_images" type="List" content="File"
            desc="Stat images from an F-contrast." />
        <input name="contrasts" type="Any" desc="Stat images from an
            F-contrast." />
        <input name="output_directory" type="Directory" desc="The destination
            folder." />
        <output name="nii_con_images" type="List" content="File"
            desc="Contrast images from a t-contrast." />
        <output name="nii_spmT_images" type="List" content="File"
            desc="Stat images from a t-contrast." />
        <output name="nii_ess_images" type="List" content="File"
            desc="Contrast images from an F-contrast." />
        <output name="nii_spmF_images" type="List" content="File"
            desc="Stat images from an F-contrast." />
    </unit>
    """
    nii_con_images = []
    nii_spmT_images = []
    nii_ess_images = []
    nii_spmF_images = []
    for images, nii_images in [(con_images, nii_con_images),
                               (spmT_images, nii_spmT_images),
                               (ess_images, nii_ess_images),
                               (spmF_images, nii_spmF_images)]:
        if isinstance(images, list):
            for cnt, im_path in enumerate(images):

                # Build the output image file name
                contrast_name = contrasts[cnt][0].lower().replace(" ", "_")
                fname = "{0}_{1}.nii.gz".format(
                    os.path.basename(im_path).split(".")[0], contrast_name)
                out = os.path.join(output_directory, fname)

                # Use nibabel for the image format conversion
                image = nibabel.load(im_path)
                nibabel.save(image, out)

                # Save destination path
                nii_images.append(out)

    return nii_con_images, nii_spmT_images, nii_ess_images, nii_spmF_images
