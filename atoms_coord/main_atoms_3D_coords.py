import argparse
from atoms_coordinates import Bionoi_coord
import os
import skimage
from skimage.io import imshow

from skimage.transform import rotate
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',
                        default="./E004921-binding-residue.mol2",
                        required=False,
                        help='the protein/ligand mol2 file')
    parser.add_argument('-out',
                        default="./output/",
                        required=False,
                        help='the folder of output file')

    parser.add_argument('-proDirect',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        required=False,
                        help='the direction of projection(from all, xy+,xy-,yz+,yz-,zx+,zx-)')

    return parser.parse_args()


def gen_output_filename_list(dirct):
    f_p_list = []
    if dirct == 0:
        f_p_list = ['XOY+', 'XOY-', 'YOZ+', 'YOZ-', 'ZOX+', 'ZOX-']
    return f_p_list


if __name__ == "__main__":
    args = getArgs()
    proDirect = 0

    mol = args.mol
    out_folder = args.out
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    proDirect = args.proDirect

    f_p_list = gen_output_filename_list(proDirect)
    len_list = len(f_p_list)
    proj_img_list = []


    # ===================================== Projection ===============================================

    for i in range(len_list):
        atoms_coords = Bionoi_coord(mol=mol,
                                 bs_out=out_folder+f_p_list[i],
                                 proDirct=i+1)


