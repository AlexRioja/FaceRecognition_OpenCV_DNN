# FaceRecognition with DeepNeuralNetworks and OpenCV
This is a Face Recognition system using Deep Neural Networks with the help of OpenCV
It's a complete system itself as it lets you create the dataset, train the dnn with it and then perform the recognition in a live stream capture video.

It's pretty reliable and can be used as a security tool

### Considerations

TESTED on UBUNTU 19.04

You may have to change some path parameters on Windows Systems!

## USAGE :

Execute the operations in this order:
```
create_datasets.py -l <dataset_name> 
```
this will create the dataset in resources/faces_2_recognize. Then execute:

```
extract_embed_features.py
```
that will extract the main features from the dataset and prepare them to be the trainnig data for the algorithm.
Then execute :
```
train_model.py
```
(YOU WILL NEED AT LEAST 2 DATASETS IN faces_2_recognize FOLDER IN ORDER FOR THIS TO WORK!).
And then enjoy executing:

```
recognize_video.py 
```
## How does it work? :

With the dataset features extracted, a Deep Neural Network is trained (SVC (Support Vector Classification) from scikit, that is a type of SVM (Support Vector Machine), a supervised learning method used for classification, in this case image classification) with the extracted features and then...voilà!.
We can pass images to the DNN and it will tell us the prediction of where does the DNN put this image (in what classification).

## TO-DOs

* Implement algorithm to align faces (the prediction works better if the face is correctly aligned)
* Create Script that works as a Trigger for others retrieving face identification
## Authors

**Alejandro Martinez** 


## License

This project is licensed under the MIT License - see [MIT License](https://opensource.org/licenses/mit-license.php) 

