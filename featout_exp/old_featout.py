import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import convolve2d
from featout_exp import IMAGESIZE
from captum.attr import Saliency


def simple_gradient_saliency(net, input_img, label):
    """
    Simplest interpretable method
    Takes an image and computes the gradients
    """
    initial_input = input_img.clone()
    # activate gradients
    initial_input.requires_grad = True
    net.eval()
    # saliency method --> can use other method here
    saliency = Saliency(net)
    grads = saliency.attribute(initial_input, target=label)
    return grads


def get_max_activation(gradients, filter_size=3):
    """
    Get the coordinates where the activation is maximal
    Includes smoothing with an all-ones filter of size filter_size
    """
    grads_mean = np.mean(gradients, axis=0)
    # smooth the results to avoid using outlier activation
    filtered = convolve2d(
        grads_mean,
        np.ones((filter_size, filter_size)),
        mode="same",
    )
    # get max of smoothed array
    max_act = np.argmax(filtered.flatten())
    # get corresponding x and y coordinates
    max_x = max_act // grads_mean.shape[1]
    max_y = max_act % grads_mean.shape[1]
    return max_x, max_y
    
def zero_out(img, max_coordinates, patch_radius=4):
    """
    Zero out a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()
    modified_input[:, :, max_x - patch_radius:max_x + patch_radius + 1,
                   max_y - patch_radius:max_y + patch_radius + 1] = 0
    return modified_input

def transform_cifar(img):
    """for cifar, we need to transform the images"""
    return np.transpose(
        (img.cpu().detach().numpy() / 2) + 0.5, (2, 1, 0)
    )

def img_to_npy(img):
    img =  img.squeeze().cpu().detach().numpy()
    img = np.transpose(img, (2,1,0))
    print( img.shape)
    return img




class Featout(torch.utils.data.Dataset):
    """
    Inspired from https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671/3
    Example usage:
    dataset = Featout(normal arguments of super dataset)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True)
    loader.dataset.start_featout(net)
    """

    def __init__(
        self, dataset, plotting_path,do_plotting = False, device = "cuda", *args, **kwargs
    ):
        """
        Args:
            dataset: torch Dataset object (must impelemnt getitem and len)
        """
        # actual dataset
        self.dataset = dataset
        # initial stage: no blurring
        self.featout = False
        # set path where to save plots (set to None if no plotting desired)
        self.plotting = plotting_path
        self.do_plotting = do_plotting
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Main workflow: Get image, if label correct, then featout (blur/zero)
        and return the modified image
        """
        # call method from base dataset
        image, label = self.dataset.__getitem__(index)
        
        # image shape [3, 254,245 ]
        # print("Curretn image shape",image.shape)
        if self.featout:
            
            in_img = torch.unsqueeze(image, 0)

            in_img = in_img.to(self.device)
            gpu_label = label.to(self.device)
            
            # run a prediction with the given model --> TODO: this can be done
            # more efficiently by passing the predicted labels from the
            # preceding epoch to this class
            _, predicted_lab = torch.max(
                self.featout_model(in_img).data, 1
            )
            # only do featout if it was predicted correctly
            if predicted_lab == label:
                # get model attention via gradient based method
                gradients = self.algorithm(
                    self.featout_model, in_img, gpu_label
                ).detach().cpu()[0].numpy()
                # Compute point of maximum activation
                max_x, max_y = get_max_activation(gradients)

                # blur patch at activation (feat-out)
                blurred_image = self.blur_method(
                    in_img, (max_x, max_y), patch_radius=4
                )
                # save images before and after if plotting is desired
                if self.do_plotting == True:
                    new_grads = self.algorithm(
                        self.featout_model,
                        blurred_image,
                        label,
                    )[0].numpy()
                    plot_together(
                        image,
                        gradients,
                        blurred_image[0],
                        new_grads
                    )

                image = blurred_image[0]

        return image, label

    def start_featout(
        self,
        model,
        blur_method=zero_out,
        algorithm=simple_gradient_saliency,
    ):
        """
        We can set here whether we want to blur or zero and what gradient alg
        """
        print("\n STARTING FEATOUT \n ")
        self.featout = True
        self.algorithm = algorithm
        self.featout_model = model
        self.blur_method = blur_method

    def stop_featout(self):
        self.featout = False





def get_overlayed_img(image, gradients):
    """
    Normalize gradients and overlay image with them (red channel)
    """
    normed_gradients = np.mean(gradients, axis=0)
    normed_gradients = (
        normed_gradients - np.min(normed_gradients)
    ) / (
        np.max(normed_gradients) - np.min(normed_gradients)
    )
    # Take image in greyscale
    transformed = transform_cifar(image)
    overlayed = np.tile(
        np.expand_dims(
            np.mean(transformed, axis=2).copy(), 2
        ),
        (1, 1, 3),
    )
    # colour the gradients red
    overlayed[:, :, 0] = normed_gradients
    return overlayed


def plot_together(
    image,
    gradients,
    blurred_image,
    new_grads,
    save_path="outputs/test.png",
):
    """
    Plot four images: the original one, then overlayed by gradients, then the
    blurred one, then this one overlayed by the new gradients
    """
    # get the points of max activation
    max_x, max_y = get_max_activation(gradients)
    new_max_x, new_max_y = get_max_activation(new_grads)
    # Make figure
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(img_to_npy(image))
    plt.title("Original input image")
    plt.subplot(1, 4, 2)
    plt.imshow(get_overlayed_img(image, gradients))
    plt.title(
        f"Model attention BEFORE blurring (max at x={max_x}, y={max_y})"
    )
    plt.subplot(1, 4, 3)
    plt.imshow(img_to_npy(blurred_image))
    plt.title("Modified input image (blurred)")
    plt.subplot(1, 4, 4)
    plt.imshow(get_overlayed_img(blurred_image, new_grads))
    plt.title(
        f"Model attention AFTER blurring (max at x={max_x}, y={max_y})"
    )
    plt.show()

