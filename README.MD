# IICCSSS Hackathon

## Stop FREAKOUT instad FEATOUT
In optimizing we mainl focused on the Dataset and  the way the most relevant gradients representing the attention of the model were chosen and the blurring of the image. 
We settled for the change of the technique as we hoped taking out the most relevant features in each epoch would force the model to consider all aspects of the given image. 

### Data
One big part of our project was to actually find, or rather subset, a suitable dataset. As the pitched goal of the featout method was increase accuracy where a "usual" model would fail, we selected the images using a "usual" model. 
Because of the varaibility of images we chose the cats versus dogs dataset from Microsoft.
The data is loaded, wrangled and saved in two halves  [here](dataset.ipynb) file. One half will be used for training ad one for testing, and vice versa. For simplicit we only use two fold cross validation, since we don't want to evaluate the model but to gain insights on how the images themselves are classified by the model.
The data is then loaded [here](load_data.ipynb)

We then trained a MobileNetV3 Model of pytorch to classify the data. For each image in the test dataset we obtained a loss value to be able to rank the images. We took the images with the highest loss to build our own dataset, as these images were the images the "usual" model would fail to predict correctly. 
This way we make sure that we have a small and for computers dfficult to understand dataset.
Analysis of the images as well as the worst and best images classified images can be found in [Analysis](analyse_worst.ipynb)  and [Bad Pictures](visually_worst.ipynb).


### Technique
We made one major change of the technique, which then were improved by two smaller changes.
The major change was the number of attention-patches that were taken into account and the way these patches were blurred out.
We did not only take highset gradient, representing the most "attention" of the model towards the point, into account but as a standard the three highest gradient patches.
We iteratively select the highest gradient and "block" the patch of size denset_size (in our case 10% of the image size) round it so our next highest gradient is chosen from another region of the image. 
In the basic version we chose three gradient patches and then block a circle of a certain size around it.
The process of including the filter in the model basically stayed the same, with minor changes due to dimensional shifts.
All of this is done in the [Featout Main File](MJ_featout_jupyter.ipynb).


        
