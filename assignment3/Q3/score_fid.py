import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
import numpy as np
from classify_svhn import Classifier

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
    # testset_feature_iterator is the Target distribution 'p'
    # sample_feature_iterator is The sample distribution  'q'

        #######################

    #   1. Get mu and cov for both distribution

        ######################

    sample_features  = []
    testset_features = []

    print('before iter')

    for i, data in enumerate(sample_feature_iterator):

        sample_features += [data]

    for i, data in enumerate(testset_feature_iterator):
        testset_features  += [data]
    print('after iters')
    sample_features = np.asarray(sample_features, dtype=np.float64)
    mu_sample = np.mean(sample_features, axis=0, dtype=np.float64)
    cov_sample = np.cov(sample_features)


    testset_features = np.asarray(testset_features, dtype=np.float64)
    mu_test = np.mean(testset_features, axis=0, dtype=np.float64)
    cov_test = np.cov(testset_features)
    
        #######################

    #   2. Calculate FID ---> d2((μ_p,Σ_p),(μ_q,Σ_q))=||μ_p −μ_q||^2 +Tr(Σ_p +Σ_q −2(Σ_p Σ_q)^{1/2})
    #                                where p is testset and q is sample
        ######################

    # First term
    L2_mu_norm = mu_sample - mu_test

    # Second and third terms (inside Tr)
    sample_trace = np.trace(cov_sample)
    test_trace = np.trace(cov_test)

    # Third term
    cov_mul  = np.matmul(cov_sample, cov_test)
    temp = np.identity(cov_mul.shape[0])*5e-10
    # Imaginary bit removal
    eps = np.identity(cov_mul.shape[0])*5e-10
    cov_mul = scipy.linalg.sqrtm(cov_mul + eps)
    trace_p = np.trace(cov_mul)

    return L2_mu_norm + sample_trace + test_trace + 2*(trace_p)


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
