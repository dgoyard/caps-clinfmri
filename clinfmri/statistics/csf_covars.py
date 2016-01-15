#! /usr/bin/env python
##########################################################################
# NSAP - Copyright (C) CEA, 2015
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import os
import numpy as np
import nibabel
from scipy.ndimage.morphology import binary_erosion
import scipy.io

# Nipy import
import nipy
from nipy.algorithms.resample import resample
from nipy.core.reference import coordinate_map as cmap
from nipy.core.reference.coordinate_system import CoordSysMaker

# Clindmri import
from clindmri.segmentation.freesurfer import mri_convert

# Matplotlib import
import matplotlib.pyplot as plt


def xyz_affine(big_aff, xyz=[0, 1, 2], verbose=0):
    """ Select the xyz part of an affine transform.

    Parameters
    ----------
    big_affine: array (N, N)
        a N-1d affine matrix, N >= 4.
    xyz: list of int (optional, default [0, 1, 2])
        the x, y, z indices used to construct the desired affine matrix.
    verbose: int (optional, default 0)
        the verbosity level.

    Returns
    -------
    affine: array (4, 4)
        the desired 3d affine matrix.
    """
    # Select the matrix components of insterest
    affine = big_aff[xyz, :][:, xyz]

    # Get the associated translation
    trans = big_aff[xyz, -1].reshape((affine.shape[0], 1))

    # Contruct the final affine transformation for homogeneous coordianates
    last_row = np.zeros((1, affine.shape[1] + 1))
    last_row[0, -1] = 1.
    affine = np.hstack((affine, trans))
    affine = np.vstack((affine, last_row))
    if verbose:
        print("\nold affine:\n", big_aff, "\nnew affine:\n", affine)

    return affine


def resample_image(source_file, target_file, output_directory,
                   w2wmap_file=None, erode_path_nb=0, order=3,
                   cval=0.0, verbose=0):
    """ Resample the source image to match the target image using Nipy.

    Parameters
    ----------
    source_file: str (mandatory)
        the image to resample.
    target_file: str (mandatory)
        the reference image.
    outdir: str (mandatory)
        the folder where the resampled image will be saved.
    w2wmap: array (4, 4) or callable
        physical to physical transformation.
    erode_path_nb: Int (optional, default 0)
        the number of path of the erosion. Performed before resampling.
    verbose: int (optional, default 0)
        the verbosity level.

    Returns
    -------
    resampled_file: str
        the resampled image.

    CAPSUL header
    -------------
    <unit>
        <input name="source_file" type="File" desc="the image to resample."/>
        <input name="target_file" type="File" desc="the reference image."/>
        <input name="output_directory" type="Directory" desc="the folder where
            the resampled image will be saved."/>
        <input name="w2wmap_file" type="File" desc="physical to physical
            transformation file."/>
        <input name="erode_path_nb" type="Int" desc="the number of path in
            erosion of the mask"/>
        <input name="order" type="Int" desc="interpolation mode, 0 = nearest
            neighbour"/>
        <input name="cval" type="Float" desc=""/>
        <input name="verbose" type="Int" desc="verbosity level"/>
        <output name="resampled_file" type="File" desc="the resampled image."/>
    </unit>

    """
    # get world to world transformation
    # SPM version
    w2wmap = scipy.io.loadmat(w2wmap_file)["Affine"]
    # fsl version
#    w2wmap = np.fromfile(w2wmap_file, sep=" ")
#    w2wmap = w2wmap.reshape(4, 4)

    w2wmap = np.linalg.inv(w2wmap)

    # Get target image information
    target_image = nipy.load_image(target_file)
    onto_shape = target_image.shape[:3]
    onto_aff = xyz_affine(target_image.affine, xyz=[0, 1, 2], verbose=verbose)

    # Define index and physical coordinate systems
    arraycoo = "ijklmnopq"[:len(onto_shape)]
    spacecoo = "xyztrsuvw"[:len(onto_shape)]
    if verbose > 0:
        print("\narraycoo: ", arraycoo, "\nspacecoo: ", spacecoo,
              "\nonto_aff\n", onto_aff)
    dmaker = CoordSysMaker(arraycoo, 'generic-array')
    rmaker = CoordSysMaker(spacecoo, 'generic-scanner')
    cm_maker = cmap.CoordMapMaker(dmaker, rmaker)
    cmap_out = cm_maker.make_affine(onto_aff)
    if verbose > 0:
        print("cmap_out:\n", cmap_out)

    # Define the default physical to physical transformation
    if w2wmap is None:
        w2wmap = np.eye(onto_aff.shape[0])
    if verbose > 0:
        print("w2wmap:\n", w2wmap)

    # erode anatomic mask if requestd
    if erode_path_nb > 0:
        # get anatomic mask
        source_image = nibabel.load(source_file)
        source_data = source_image.get_data()

        eroded_image = binary_erosion(
            source_data,
            iterations=erode_path_nb).astype(source_data.dtype)

        # save
        _temp = nibabel.Nifti1Image(eroded_image, source_image.get_affine())
        source_file = os.path.join(output_directory,
                                   'eroded_anat_mask.nii.gz')
        nibabel.save(_temp, source_file)

    # Get eroded anatomic mask
    source_image = nipy.load_image(source_file)

    # resample
    resampled_image = resample(
        source_image, cmap_out, w2wmap, onto_shape, order=order, cval=cval)

    # save
    resampled_file = os.path.join(
        output_directory,
        "resampled_{0}".format(os.path.basename(source_file)))
    nipy.save_image(resampled_image, resampled_file)

    return resampled_file


def get_covars(csfmask_file, func_file, min_nb_of_voxels=20, nb_covars=5,
               verbose=0, output_directory=None):
    """ Compute covariates that represent the CSF variability in a functional
    timeserie.

    Parameters
    ----------
    csfmask_file: str (mandatory)
        a binary mask of the CSF in the functional space.
    func_file: str (mandatory)
        a functional volume of size (X, Y, Z, T).
    min_nb_of_voxels: int (optional, default 50)
        the criterion used to select a CSF ROI with specific size.
    nb_covars: int (optional, default 5)
        the number of covariates used to explain the CSF variability.
    verbose: int (optional, default 0)
        the verbosity level.
    output_directory: str (optional, default None)
        for debuging purpose: if the verbosity level is > 1 save the mask used
        to select the functional time series.

    Returns
    -------
    covars: array (T, nb_covars)
        the requested number of covariates that represent the CSF variability.

    CAPSUL HEADER
    -------------

    <unit>
        <input name="csfmask_file" type="File"
            desc="a binary mask of the CSF in the functional space."/>
        <input name="func_file" type="File"
            desc="a functional volume of size (X, Y, Z, T)."/>
        <input name="min_nb_of_voxels" type="Int"
            desc="the criterion used to select a CSF ROI with specific size.
            Optional (default=50)"/>
        <input name="nb_covars" type="Int"
        desc="the number of covariates used to explain the CSF variability.
        Optional (default=5)"/>
        <input name="verbose" type="Int" desc="the verbosity level.
            Optional (default=0)"/>
        <input name="output_directory" type="Directory"
            desc="for debuging purpose: if the verbosity level is > 1
                  save the mask used
                  to select the functional time series."/>

        <output name="covars" type="File" desc="the requested number of
            covariates that represent the CSF variability."/>
    </unit>
    """
    # Erode the mask until we have N < min_nb_of_voxels ones
    csf_image = nibabel.load(csfmask_file)
    csf_array = csf_image.get_data()
    csf_array[np.where(csf_array != 0)] = 1
    if len(np.where(csf_array == 1)[0]) > min_nb_of_voxels:
        while True:
            csf_tmp_array = binary_erosion(csf_array, iterations=1)
            nb_of_ones = len(np.where(csf_tmp_array == 1)[0])
            if nb_of_ones < min_nb_of_voxels:
                break
            csf_array = csf_tmp_array
    else:
        raise ValueError(
            "Not enough CSF voxels in mask '{0}'.".format(csfmask_file))
    if verbose > 1:
        csf_mask = nibabel.Nifti1Image(csf_array.astype(int),
                                       csf_image.get_affine())
        nibabel.save(csf_mask, os.path.join(output_directory,
                                            "covars_mask.nii.gz"))

    # Compute a SVD
    func_array = nibabel.load(func_file).get_data()
    csftimeseries = func_array[np.where(csf_array == 1)].T
    csftimeseries = csftimeseries.astype(float)
    csftimeseries -= csftimeseries.mean(axis=0)
    u, s, v = np.linalg.svd(csftimeseries, full_matrices=False)
    if verbose > 2:
        plt.plot(s)
        plt.show()

    # Get the covariates that represent the CSF variability
    covars = u[:, :nb_covars]

    np.savetxt(os.path.join(output_directory, "covars.txt"), covars)
    covars = os.path.join(output_directory, "covars.txt")

    return covars


def complete_regressors_file(input_file, covars_file, output_directory,
                             add_extra_mvt_reg=False):
    """
    Complete the rp files with covars from an extra file

    CAPSUL HEADER
    -------------

    <unit>
        <input name="input_file" type="File" desc="the current regressors
            file"/>
        <input name="covars_file" type="File" desc="the regressors to add file"
            />
        <input name="output_directory" type="Directory" desc="the directory
            that will contain the output file"/>
        <input name="add_extra_mvt_reg" type="Bool" desc="Command wether the
            movement parameters need to be completed (t2, t3, t-1, t+1 will
            be added)"/>
        <output name="covars" type="File" desc="the completed covars file"/>
    </unit>
    """
    covars = np.loadtxt(covars_file)
    # if we don't want any extra noise regressor (nb_extra_covars = 0), the
    # covars value here will be an empty array.

    rp = np.loadtxt(input_file)

    if add_extra_mvt_reg:
        # Add translation square and cube, plus shift
        t, r = rp[:, :3], rp[:, 3:]
        kl = np.vstack((t[:1, :], t[:-1, :]))
        ke = np.vstack((t[1:, :], t[-1:, :]))
        if covars.shape[0] == 0:
            out = np.column_stack((t, r, t**2, t**3, ke, kl))
        else:
            out = np.column_stack((t, r, t**2, t**3, ke, kl, covars))
    else:
        if covars.shape[0] != 0:
            out = np.column_stack((rp, covars))
        else:
            out = rp

    # normalize
    out = (out - out.mean(axis=0)) / out.std(axis=0)
    covars = os.path.join(output_directory, "complete_reg_file.txt")
    np.savetxt(covars, out, fmt="%5.8f")
    return covars
