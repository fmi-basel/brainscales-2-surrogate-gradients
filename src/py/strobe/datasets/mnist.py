import os

import numpy as np
import scipy
from scipy.ndimage import zoom
from PIL import Image

import torch
import torch.utils.data
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets.mnist import read_label_file, read_image_file, download_and_extract_archive

def only_zoom(images):
    images_zoomed = np.zeros((images.shape[0], 16, 16))
    for i in range(images.shape[0]):
        images_zoomed[i, :, :] = zoom(images[i, :, :], 16 / 28)

    return torch.from_numpy(images_zoomed.astype(np.uint8))


def crop_and_zoom(images):
    images_cropped = images[:, 2:-2, 2:-2]
    images_zoomed = np.zeros((images_cropped.shape[0], 16, 16))
    for i in range(images_cropped.shape[0]):
        images_zoomed[i, :, :] = zoom(images_cropped[i, :, :], 16 / 24)

    return torch.from_numpy(images_zoomed.astype(np.uint8))


class MNIST16x16(MNIST):
    """`16x16 pixel version of the MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
    The original data are rescaled after downloading and stored in this processed state.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def download(self):
        """Download and rescale the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        print(self.resources)
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

        # process and save as torch files
        print('Processing...')
        
        training_set = (
            crop_and_zoom(read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            crop_and_zoom(read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class FashionMNIST16x16(FashionMNIST):
    """`16x16 pixel version of the MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
    The original data are rescaled after downloading and stored in this processed state.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def download(self):
        """Download and rescale the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        
        training_set = (
            only_zoom(read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            only_zoom(read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


if __name__ == "__main__":
        
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs

    from torchvision import transforms
    from spikes import PixelsToSpikeTimes, SpikeTimesToDense, SpikeTimesToDense

    mnist = FashionMNIST16x16("data", train=True, download=True, transform=transforms.ToTensor())
    #mnist = MNIST16x16("data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=5, shuffle=True)

    to_spike = PixelsToSpikeTimes(tau=8e-6, t_max=100e-6)
    to_dense = SpikeTimesToDense(2.5e-6, 40)

    for batch, (x, y) in enumerate(train_loader):
        image = x[0]

        fig = plt.figure()
        grid = gs.GridSpec(1, 2)
        axes = [fig.add_subplot(grid[0, i]) for i in range(2)]

        axes[0].imshow(image.numpy()[0])
        axes[0].set_title(mnist.classes[y[0]])

        spikes = to_dense(to_spike(x))
        spike_train = spikes[0].reshape(spikes[0].shape[0], 256)

        axes[1].imshow(spike_train.T)

        fig.savefig("input.pdf")

        asd
