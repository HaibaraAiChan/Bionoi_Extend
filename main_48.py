import argparse
from Voronoi import Bionoi
import os
import skimage
from skimage.io import imshow

from skimage.transform import rotate
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',
                        default="./examples/4v94E.mol2",
                        required=False,
                        help='the protein/ligand mol2 file')
    parser.add_argument('-out',
                        default="./output_400/",
                        required=False,
                        help='the folder of output images file')
    parser.add_argument('-dpi',
                        default=256,
                        required=False,
                        help='image quality in dpi')
    parser.add_argument('-size', default=256,
                        required=False,
                        help='image size in pixels, eg: 128')
    parser.add_argument('-alpha',
                        default=1.0,
                        required=False,
                        help='alpha for color of cells')
    parser.add_argument('-colorby',
                        default="residue_type",
                        choices=["atom_type", "residue_type", "residue_num"],
                        required=False,
                        help='color the voronoi cells according to {atom_type, residue_type, residue_num}')
    parser.add_argument('-imageType',
                        default=".jpg",
                        choices=[".jpg", ".png"],
                        required=False,
                        help='the type of image {.jpg, .png}')
    parser.add_argument('-proDirect',
                        type=int,
                        default=4,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        required=False,
                        help='the direction of projection(from all, xy+,xy-,yz+,yz-,zx+,zx-)')
    parser.add_argument('-rotAngle2D',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        required=False,
                        help='the angle of rotation(from original all, 0, 90, 180, 270)')
    parser.add_argument('-flip',
                        type=int,
                        default=0,
                        choices=[0, 1, 2],
                        required=False,
                        help='the type of flipping(all, original, up-down)')
    parser.add_argument('-save_fig',
                        default=False,
                        choices=[True, False],
                        required=False,
                        help='whether the original image needs save (True, False)')
    return parser.parse_args()


def gen_output_filename_list(dirct, rotAngle, flip):
    f_p_list = []
    f_r_list = []
    f_f_list = []

    if dirct != 0:
        name = ''
        if dirct == 1:
            name = 'XOY+'
        elif dirct == 2:
            name = 'XOY-'
        elif dirct == 3:
            name = 'YOZ+'
        elif dirct == 4:
            name = 'YOZ-'
        elif dirct == 5:
            name = 'ZOX+'
        elif dirct == 6:
            name = 'ZOX-'
        f_p_list.append(name)
    elif dirct == 0:
        f_p_list = ['XOY+', 'XOY-', 'YOZ+', 'YOZ-', 'ZOX+', 'ZOX-']

    if rotAngle != 0:
        name = ''
        if rotAngle == 1:
            name = '_r0'
        elif rotAngle == 2:
            name = '_r90'
        elif rotAngle == 3:
            name = '_r180'
        elif rotAngle == 4:
            name = '_r270'
        f_r_list.append(name)
    else:
        f_r_list = ['_r0', '_r90', '_r180', '_r270']

    if flip != 0:
        name = ''
        if flip == 1:
            name = '_OO'
        elif flip == 2:
            name = '_ud'

        f_f_list.append(name)
    else:
        f_f_list = ['_OO', '_ud']

    return f_p_list, f_r_list, f_f_list


if __name__ == "__main__":
    args = getArgs()
    proDirect = 0

    mol = args.mol
    out_folder = args.out
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    size = args.size
    dpi = args.dpi
    alpha = args.alpha
    imgtype = args.imageType
    colorby = args.colorby
    proDirect = args.proDirect
    rotAngle = args.rotAngle2D
    flip = args.flip

    f_p_list,f_r_list,f_f_list = gen_output_filename_list(proDirect,rotAngle,flip)
    len_list = len(f_p_list)
    proj_img_list = []
    rotate_img_list = []
    flip_img_list = []

    # ===================================== Projection ===============================================
    if proDirect != 0:
        atoms, vor, img = Bionoi(mol=mol,
                                 bs_out=out_folder+f_p_list[0],
                                 size=size,
                                 dpi=dpi,
                                 alpha=alpha,
                                 colorby=colorby,
                                 proDirct=proDirect)
        imshow(img)
        proj_img_list.append(img)
    else:
        for i in range(len_list):
            atoms, vor, img = Bionoi(mol=mol,
                                     bs_out=out_folder+f_p_list[i],
                                     size=size,
                                     dpi=dpi,
                                     alpha=alpha,
                                     colorby=colorby,
                                     proDirct=i+1)
            proj_img_list.append(img)
    # ---------------------------------- rotate -----------------------------------------

    col = proj_img_list
    m = len(col)

    for i in range(m):
        img = col[i]
        if rotAngle == 0:

            rotate_img_list.append(img)
            rotate_img_list.append(rotate(img, angle=90))
            rotate_img_list.append(rotate(img, angle=180))
            rotate_img_list.append(rotate(img, angle=270))
        elif rotAngle == 1:
            rotate_img_list.append(rotate(img, angle=0))
        elif rotAngle == 2:
            rotate_img_list.append(rotate(img, angle=90))
        elif rotAngle == 3:
            rotate_img_list.append(rotate(img, angle=180))
        elif rotAngle == 4:
            rotate_img_list.append(rotate(img, angle=270))
    # ---------------------------------- flip  -----------------------------------------
    # len_r = len(rotate_img_list)

    for i in range(len(rotate_img_list)):
        img = rotate_img_list[i]
        if flip == 0:
            flip_img_list.append(img)
            flip_img_list.append(np.flipud(img))

        if flip == 1:
            flip_img_list.append(img)
        if flip == 2:
            img = np.flipud(img)
            flip_img_list.append(img)


    assert len(proj_img_list) == len(f_p_list)
    assert len(rotate_img_list) == len(f_r_list)*len(f_p_list)
    assert len(flip_img_list) == len(f_f_list)*len(f_r_list)*len(f_p_list)
    filename_list = []

    fileList = list(os.listdir(out_folder))
    for file in fileList:
        if os.path.isfile(file):
            os.remove(file)

    for i in range(len(f_p_list)):
        for j in range(len(f_r_list)):
            for k in range(len(f_f_list)):
                tmp = f_p_list[i] + f_r_list[j] + f_f_list[k]
                saveFile = os.path.join(out_folder, f_p_list[i] + f_r_list[j] + f_f_list[k] + imgtype)
                filename_list.append(saveFile)

    assert len(filename_list) == len(flip_img_list)
    for i in range(len(filename_list)):
        imshow(flip_img_list[i])
        skimage.io.imsave(filename_list[i], flip_img_list[i])
