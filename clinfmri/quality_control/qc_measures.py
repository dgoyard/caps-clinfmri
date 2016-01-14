# -*- coding: utf-8 -*-

import numpy as np
import nibabel
import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib backend
mpl.use("AGG")


def power_scores(gm_mask, before_fmri_file, mtv_corrected_fmri_file,
                 after_fmri_file, fd_file,
                 output_directory, verbose=0):
    """
    Generate QC scores and jpeg images
    from Power 2014

    CAPSUL header
    -------------
    <unit>
        <input name="gm_mask" type="File" desc="the gray matter mask to select
            relevant signal"/>
        <input name="before_fmri_file" type="File" desc="the fmri file before
            all noise corrections"/>
        <input name="mtv_corrected_fmri_file" type="File" desc="the fmri file
            after movements noise correction"/>
        <input name="after_fmri_file" type="File" desc="the fmri file after
            all noise corrections"/>
        <input name="fd_file" type="File" desc="the frame-displacement file
            as outputed by the qap library"/>
        <input name="output_directory" type="Directory" desc="the directory
            that will contain the jpeg file"/>
        <input name="verbose" type="Int" desc="the verbosity level"/>
        <output name="qc_image" type="File" desc="the qc image generated
            from displacement file and gm voxel timecourses"/>
    </unit>
    """

    # define plot
    fig = plt.figure()

    # plot frame displacement
    fd_values = np.loadtxt(fd_file)
    ax = fig.add_subplot(4, 1, 1)
    plt.plot(fd_values)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylabel("FD")
    plt.title('Grey matter voxels timecourses')

    # load gray matter file
    mask_file = nibabel.load(gm_mask)
    mask_data = mask_file.get_data()

    if verbose > 0:
        print "{0} pixels will be tracked".format(np.sum(mask_data))

    mask_data = mask_data.ravel()

    # load data
    before_data_array = nibabel.load(before_fmri_file).get_data()

    # get voxels timecourse
    before_img = []
    for volume_nb in range(before_data_array.shape[-1]):
        vol = before_data_array[:, :, :, volume_nb].ravel()
        vol = vol[mask_data == 1]
        before_img.append(vol)

    # transform in numpy array and transpose
    before_img_array = np.asarray(before_img)
    before_img_array = np.transpose(before_img_array)

    # normalize each timecourse individually
    before_normed_img = []
    for index in range(before_img_array.shape[0]):
        line = before_img_array[index, :]
        _min = np.min(line)
        _max = np.max(line)
        line_normed = (255.0 * (line - _min) / (_max - _min)).astype(np.uint8)
        before_normed_img.append(line_normed)

    # transform back into numpy array
    before_normed_img = np.asarray(before_normed_img)

    # save image in main QC figure
    ax = fig.add_subplot(4, 1, 2)
    plt.imshow(before_normed_img, cmap="gray", aspect="auto")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylabel("uncorrected")

    # save voxels timecourse separately
    before_img_png = Image.fromarray(before_normed_img, 'L')
    before_img_png.save(os.path.join(output_directory, "before.png"), "PNG", )

    before_img = os.path.join(output_directory, "before.png")

    # same operations with the image AFTER movements noise corrections
    mvt_after_data_array = nibabel.load(mtv_corrected_fmri_file).get_data()

    mvt_after_img = []

    for volume_nb in range(mvt_after_data_array.shape[-1]):
        vol = mvt_after_data_array[:, :, :, volume_nb].ravel()
        vol = vol[mask_data == 1]
        mvt_after_img.append(vol)

    mvt_after_img_array = np.asarray(mvt_after_img)

    mvtafter_normed_img = []
    mvt_after_img_array = np.transpose(mvt_after_img_array)

    for index in range(mvt_after_img_array.shape[0]):
        line = mvt_after_img_array[index, :]
        _min = np.min(line)
        _max = np.max(line)
        line_normed = (255.0 * (line - _min) / (_max - _min)).astype(np.uint8)
        mvtafter_normed_img.append(line_normed)

    mvtafter_normed_img = np.asarray(mvtafter_normed_img)

    # same operations with the image AFTER all noise corrections
    after_data_array = nibabel.load(after_fmri_file).get_data()

    after_img = []

    for volume_nb in range(after_data_array.shape[-1]):
        vol = after_data_array[:, :, :, volume_nb].ravel()
        vol = vol[mask_data == 1]
        after_img.append(vol)

    after_img_array = np.asarray(after_img)

    after_normed_img = []
    after_img_array = np.transpose(after_img_array)

    for index in range(after_img_array.shape[0]):
        line = after_img_array[index, :]
        _min = np.min(line)
        _max = np.max(line)
        line_normed = (255.0 * (line - _min) / (_max - _min)).astype(np.uint8)
        after_normed_img.append(line_normed)

    after_normed_img = np.asarray(after_normed_img)

    # plot images
    ax = fig.add_subplot(4, 1, 3)
    plt.imshow(mvtafter_normed_img, cmap="gray", aspect="auto")
    ax.set_ylabel("R, T, TT,\n TTT, T-1, T+1")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
#    ax.set_xlabel("volumes")

    ax = fig.add_subplot(4, 1, 4)
    plt.imshow(after_normed_img, cmap="gray", aspect="auto")
    ax.set_ylabel("R, T, TT,\n TTT, T-1, T+1,\n WM(5), CSF(5)")
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlabel("volumes")

    # save individual image
    after_img_png = Image.fromarray(after_normed_img, 'L')
    after_img_png.save(os.path.join(output_directory, "before.png"), "PNG", )
    after_img = os.path.join(output_directory, "before.png")

    # generate QC file and return
    plt.savefig(os.path.join(output_directory, "qc_Power.png"))
    qc_image = os.path.join(output_directory, "qc_Power.png")

    fig.tight_layout()

    return qc_image, fig


def mask_overlay_image(base_image, overlay_image, n_snap, outfile, title=""):

    # get image data
    base_array = nibabel.load(base_image).get_data()
    overlay_data = nibabel.load(overlay_image).get_data()

    # init numpy array with first slices
    images = base_array[0, :, :]
    overlays = overlay_data[0, :, :]

    for index in np.arange(0, base_array.shape[0],
                           base_array.shape[0] / n_snap):
        # put all images in one row
        overlay = overlay_data[index, :, :]
        if np.sum(overlay) == 0:
            continue
        images = np.column_stack((images, base_array[index, :, :]))
        overlays = np.column_stack((overlays, overlay))

    # remove first slices
    images = images[:, base_array.shape[2]:]
    overlays = overlays[:, base_array.shape[2]:]

    # build figure
    fig, ax = plt.subplots()
    ax.imshow(images, cmap=plt.cm.gray,
              interpolation='nearest', origin='lower')
    overlays = np.ma.array(overlays, mask=overlays == 0)
    ax.imshow(overlays, cmap='bwr',
              interpolation='nearest', origin='lower')
    if len(title) > 0:
        plt.title(title)
    plt.savefig(outfile)

    return outfile, fig
