"""
Created on Mon Jan 15 13:31:59 2018
@author: chinwei

do `pip install academictorrents` before running this script
"""

import pickle
import gzip
import os
import sys
import numpy as np
import academictorrents as at

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist'],
                        help='dataset name')
    parser.add_argument('--savedir', type=str, default='datasets',
                        help='directory to save the dataset')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'mnist':
        local_filename = at.get("323a0048d87ca79b68f12a6350a57776b6a3b7fb")
        
        if sys.version_info < (3,0):
            tr, va, te = pickle.load(gzip.open(local_filename, 'r'))
        else:
            tr, va, te = pickle.load(gzip.open(local_filename, 'r'), 
                                     encoding='latin1')
        
        os.remove(local_filename)
        np.save("{}/mnist".format(args.savedir), (tr, va, te))
