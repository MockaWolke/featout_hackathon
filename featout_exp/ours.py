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
from skimage.draw import disk


def get_circle_indices(shape, center, radius):
    y, x = disk(center, radius, shape=shape)
    return y, x


# get centers of clusters
# TODO : varbalize (tempreture) accourding to training progress or certainty ?
# TODO : generelize the density calculations
def get_clustered_activations(
    gradients,
    filter_size=3,
    densety_factor=4,
    centers_count=3,
    cluster_shape_selector=get_circle_indices,
):
    # smoothen
    grads_mean = np.mean(gradients, axis=0)
    kernel = np.ones((filter_size, filter_size))
    # smooth the results to avoid using outlier activation
    filtered = convolve2d(
        grads_mean,
        kernel,
        mode="same",
    )

    centers = []
    mask = np.ones(grads_mean.shape)
    # simple way TODO : more sophisticate way without increasing computation by much
    for _ in range(0, centers_count):
        # get max of smoothed array
        max_act = np.argmax((filtered * mask).flatten())
        # get corresponding x and y coordinates
        max_x = max_act // grads_mean.shape[1]
        max_y = max_act % grads_mean.shape[1]

        centers.append((max_x, max_y))
        covered_area = cluster_shape_selector(
            mask.shape, (max_x, max_y), densety_factor
        )
        mask[covered_area[0], covered_area[1]] = 0

    return centers, mask


def gaussian_blur(img, kernel_size):
    pass



# bluring function
def blur_featurs(
    img,
    gradients,
    centers,
    intencity=0.5,
    radius_scaler=1,
    blur_method=gaussian_blur,
    shape_selector=get_circle_indices,
):
    modified_img = img.clone()
    # normalize gradients
    grads_mean = np.mean(gradients, axis=0)
    grad_abs = np.abs(grads_mean)
    grad_normalized = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min())
    # get max radius normalized to image size
    max_region_portion = 0.7
    # max_radius = int( np.sqrt( max_region_portion * torch.prod(modified_img.shape)  / np.pi ) / len(centers) * radius_scaler)
    max_radius = int(
        np.sqrt(
            max_region_portion
            * grad_normalized.shape[-1]
            * grad_normalized.shape[-2]
            / np.pi
        )
        / len(centers)
        * radius_scaler
    )

    # max_radius = int( np.sqrt( max_region_portion * np.prod(grad_normalized.shape[2:])  / np.pi ) / len(centers) * radius_scaler)

    # blur around the centers of attention/deatures
    for center in centers:
        # fit the size of the redius for bluring propotional to the magnitud of the gradient of the center
        radius = max_radius * grad_normalized[center[0], center[1]]

        # y , x = shape_selector(modified_img.shape[2:], center, radius)
        # modified_img[:,:,y,x] = 0

        blur_region = shape_selector(modified_img.shape[2:], center, radius)
        modified_img[:, :, blur_region[0], blur_region[1], ] = 0

        # print(
        #     f"###### radius : {radius} , max radius : {max_radius} , grad : {grad_normalized[center[0],center[1]]} ######"
        # )

        # kernel_size = 5
        # border_value = [0]*img.shape[-1]
        # blur_region = cv2.GaussianBlur(blur_region,kernel_size,0, borderType=cv2.BORDER_CONSTANT, borderValue=border_value)
        # modified_img[:,:,y,x] = blur_region

        # modified_img[:,:,y,x] = blur_method(modified_img[:,:,y,x])

    return modified_img


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


def transform_cifar(img):
    """for cifar, we need to transform the images"""
    return np.transpose((img.cpu().detach().numpy() / 2) + 0.5, (1,2,0))


def img_to_npy(img):
    img = img.squeeze().cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
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
        self, dataset, plotting_path, do_plotting=False, device="cuda", *args, **kwargs
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
        image, og_label = self.dataset.__getitem__(index)

        original_device = image.device
        # image shape [3, 254,245 ]
        # print("Curretn image shape",image.shape)

        # print(type(image), type(og_label))

        if self.featout:
            label = torch.Tensor([og_label]).long()

            # print("first", image.device, label.device)

            in_img = torch.unsqueeze(image, 0)

            in_img = in_img.to(self.device)
            gpu_label = label.to(self.device)
            # print("second", in_img.device, gpu_label.device, gpu_label)

            # run a prediction with the given model --> TODO: this can be done
            # more efficiently by passing the predicted labels from the
            # preceding epoch to this class
            _, predicted_lab = torch.max(self.featout_model(in_img).data, 1)

            # print(predicted_lab)
            # only do featout if it was predicted correctly
            if (predicted_lab.to(label.device).squeeze() == label).all():
                # get model attention via gradient based method

                gradients = (
                    self.algorithm(self.featout_model, in_img, gpu_label)
                    .detach()
                    .cpu()[0]
                    .numpy()
                )
                
                centers_count = 1 + int( self.exp_rate * self.max_clusters)
                total_size_portion = 0.1  # TUNING PARAMETER
                densety_factor = int( np.sqrt( total_size_portion * in_img.shape[-1] * in_img.shape[-2]  / np.pi ) / centers_count)
                
                centers, mask = get_clustered_activations(gradients, 
                                                        filter_size=3,
                                                        densety_factor=densety_factor,
                                                        centers_count=centers_count,
                                                        cluster_shape_selector=get_circle_indices)
                
                blurred_image = self.blur_method(in_img, gradients, centers, intencity=0.5)
                #-----------------------
                
                
                
                # save images before and after if plotting is desired
                if self.do_plotting == True:
                    new_grads = self.algorithm(
                        self.featout_model,
                        blurred_image,
                        label,
                    )[0].numpy()
                    plot_together(image, gradients, blurred_image[0], new_grads)

                # print(type(image), image.shape, image.device)
                image = blurred_image[0]
                
                # print(type(image), image.shape,  image.device)
                
        return image.to(original_device), torch.tensor(og_label)

    def start_featout(
        self,
        model,
        blur_method= blur_featurs,
        algorithm=simple_gradient_saliency,
        exp_rate=0.1,
        max_clusters=10
    ):
        """
        We can set here whether we want to blur or zero and what gradient alg
        """
        print("\n STARTING FEATOUT \n ")
        self.featout = True
        self.algorithm = algorithm
        self.featout_model = model
        self.blur_method = blur_method
        self.exp_rate = exp_rate # TUNING PARAMETER
        self.max_clusters = max_clusters  # TUNING PARAMETER


    def stop_featout(self):
        self.featout = False


def get_overlayed_img(image, gradients):
    """
    Normalize gradients and overlay image with them (red channel)
    """
    normed_gradients = np.mean(gradients, axis=0)
    normed_gradients = (normed_gradients - np.min(normed_gradients)) / (
        np.max(normed_gradients) - np.min(normed_gradients)
    )
    # Take image in greyscale
    transformed = transform_cifar(image)
    overlayed = np.tile(
        np.expand_dims(np.mean(transformed, axis=2).copy(), 2),
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
    plt.title(f"Model attention BEFORE blurring (max at x={max_x}, y={max_y})")
    plt.subplot(1, 4, 3)
    plt.imshow(img_to_npy(blurred_image))
    plt.title("Modified input image (blurred)")
    plt.subplot(1, 4, 4)
    plt.imshow(get_overlayed_img(blurred_image, new_grads))
    plt.title(f"Model attention AFTER blurring (max at x={max_x}, y={max_y})")
    plt.show()


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