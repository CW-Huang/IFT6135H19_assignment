import argparse
import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
from scipy import linalg

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`
    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.
    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.
    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    The following link is an example of implementation of
    FID score computation in pytorch:
        https://github.com/mseitzer/pytorch-fid
    We used it has reference to make sure we were on the right track.
    For exemple the author of the repo above included way to deal with
    the (normal) presence of (very small e.g. 1e-18) imaginary
    component of the output of linalg.sqrtm that computes the square
    root of a matrix. These imaginary component comes from the fact
    that computing the square root of a matrix requires to
    do floating point arithmetic with imarinary component. So any number
    smaller than some machine-epsilon can just be interpreted as a true zero.
    """

    sample_features_list  = []
    testset_features_list = []

    # stats for samples
    for i,h in enumerate(sample_feature_iterator):
        sample_features_list += [h]

    sample_features = np.asarray(sample_features_list, dtype=np.float64)
    avg_sample = np.mean(sample_features, axis=0, dtype=np.float64)
    cov_sample = np.cov(sample_features,rowvar=False)

    # stats for test
    for i,h in enumerate(testset_feature_iterator):
        testset_features_list += [h]

    testset_features = np.asarray(testset_features_list, dtype=np.float64)
    avg_test = np.mean(testset_features, axis=0, dtype=np.float64)
    cov_test = np.cov(testset_features,rowvar=False)

    # mu should be 512, cov should be 512 x 512
    # print( "mu  shape " , avg_test.shape )
    # print( "cov shape " , cov_test.shape )

    #
    delta              = avg_sample - avg_test
    cov_sample_test, _ = linalg.sqrtm(cov_sample.dot(cov_test), disp=False)

    # imaginary component, we only care about the diagonal component
    # if they are close enough to zero, we set them to true zero
    if np.iscomplexobj(cov_sample_test): # check if it has an imaginary part (even zero)
        if not np.allclose(np.diagonal(cov_sample_test).imag, 0, atol=1e-5):
            # above this threshold, warn user
            print("Imaginary component too high in computation, might be an issue")
        cov_sample_test = cov_sample_test.real

    ##################################################

    FID_score = delta.dot(delta) + np.trace(cov_sample) + np.trace(cov_test) - 2 * np.trace(cov_sample_test)
    return FID_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
