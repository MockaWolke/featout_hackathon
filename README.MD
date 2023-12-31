# IICCSSS Hackathon

Hey (;

We just quickly updated the readme and the images. We hope that's okay. If not just, just choose an earlier commit.


![stg](imgs/porter.png)

## Stop FREAKOUT instad FEATOUT
In optimizing we mainly focused on the Dataset and some of the techniques the algorithm used. Of these techniques we primarily focused on the way the maximum gradients were chosen and how the area around the maximum gradients was blurred out. Other aspects of the algorithm, such as chosing how often the FeatOut actually takes place or other methods to derive the models attention, were not considered in this analysis.
We settled for these changes of the technique as we hoped precisely taking out out the most relevant features in each epoch would force the model to consider all aspects of the given image, witout blurring to much of the image and therefore potentially making it impossbile for the model to derive any features at all. The sophisticated modifications we made to the technique in later steps enable to strike the balance between exactly blurring the main features of interests while still enabling the model to freely consider new emerging features.

### Data
One big part of our project was to actually find, or rather subset, a suitable dataset. As the pitched goal of the featout method was increase accuracy where a "usual" model would fail, we selected the images using a "usual" model. 
Because of the varaibility of images we chose the cats versus dogs dataset from Microsoft. The dataset needs to be downloaded and unzipped from https://www.microsoft.com/en-US/download/details.aspx?id=54765
We stored all of the images in a foled called PetImages.
The data is read, wrangled and saved in two halves  [here](dataset.ipynb) file. One half will be used for training ad one for testing, and vice versa. For simplicit we only use two fold cross validation, since we don't want to evaluate the model but to gain insights on how the images themselves are classified by the model.
The data is then loaded [here](load_data.ipynb).

We then trained a MobileNetV3 Model of pytorch to classify the data. For each image in the test dataset we obtained a loss value to be able to rank the images. We took the images with the highest loss to build our own dataset, as these images were the images the "usual" model would fail to predict correctly. The training and savin of the data can be retraced with the files in the [Collab](collab_notebooks) folder.
This way we make sure that we have a small and for computers dfficult to understand dataset. 
Analysis of the images as well as the worst and best images classified images can be found in [Analysis](analyze/analyze_worst.ipynb)  and [Bad Pictures](visually_worst.ipynb).


The worst images:
![worst](imgs/first_worst.jpg)

The best images:
![best](imgs/first_best.jpg)

You can notice a stellar dataset quality (:

### Technique
We made one major change of the technique, which then were improved by two smaller changes. All of the changes build up on eachother and increase the complexity of the FeatOut Algorithm.
#### 3 clusters, Static Blur
The major change was the number of attention-patches that were taken into account and the way these patches were blurred out.
We did not only take highset gradient, representing the most "attention" of the model towards the point, into account but as a standard the three highest gradient patches.
We iteratively select the highest gradient and "block" the a patch arround it of a certain size. (in our case 10% of the image size) round it so our next highest gradient is chosen from another region of the image. 
In the basic version we chose three gradient patches and then block a circle of a certain size around it.
The process of including the filter in the model basically stayed the same, with minor changes due to dimensional shifts.
All of this is done in the [Featout Main File](MJ_featout_jupyter.ipynb).


#### 3 clusters, Blur based on gradient intensity
The one smaller change was not to zero out the entire blur patch but to actually blur it based on the intensity of the gradient. It depends on the intensity in two ways: the size of the blur is proportional to the intensity of the gradient in the patch but also the rate of the masking, or blurring, is proportional to the intensity. So if we have a high gradient we mask a bigger area around the gradient and the value in the center of the covered area is closer to zero than when we have a low gradient. 
This is supposed to ensure that it is harder for the model to heavily fixate on one are of the picture rather than taking the "bigger picture" into account and diversifying the features. 

![dsa](imgs/3_Clusters_StaticBlur_ex1.jpg)
![dsa](imgs/3_Clusters_StaticBlur_ex2.jpg)
#### Time-variant Numbers of clusters, Static Blur
The second smaller change was to make the size of the gradient patch clusters dependent on the epoch one was currently training in. The higher the epoch the more clusters would be blurred. This is implemented in a linear fashion to keep things simple. Increasing the number of clusters ensures that as, usually, the model gets more and more confident and takes less new features into account with this method it is forced to "reconsider" the used features. On a longer run this should ensure that the models taks various features into account.

All of these methods were implemented and tested on the MNST dataset and finally trained all at one with our carefully selected dataset. All these methods together should ensure that the models attention is shifted to seemingly more irrelevant features, hopefully giving it an overall better feature recognition. 

Reference images of the MNST Dataset were saved in the [Images](imgs) folder. They may not be fully available due to time issues during restructuring.

### Results
We ran (or to be more precise: are currently running!) 3 analysis. All three are run on the selected dataset of bad pictures.
The first one is using the MobileNetV3. Expectedly it is yielding a bad accuracy witha fast runtime.
The second one uses the algorithm provided by Nina. 
The third one uses the algorithm designed by us, with the cluster size increasing with time and the blur being based on gradient intensity. Due to the fact that training takes a lot of time we were unable to test all the new features of the algorithm seperately.

A comparison of all results can be found in the file [Results](analysis.ipynb).

Here the Control model is not applying any augmentatation.
The Standard is using Ninas approach and the Stronger our.
As you can see FEATOUT did not aprove the performance ): But we were clearly able to procude a very hard test dataset, with perhabs to bad image quality afterall.


![imgs](imgs/results.jpg)