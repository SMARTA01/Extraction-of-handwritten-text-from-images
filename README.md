# Extraction of handwritten-text from an image

##  Synopsis

One of the most difficult tasks in NLP is handwriting. It's because it can differ from person to person. However, certain characters (for example, English) are very similar. We use contextualized information and lexical matching as a human starting point. While humans have the ability to determine whether it is "O" or "0" from contextualised information, "O" can sometimes be written as "0." For instance, "0" will be used in phone numbers, whereas "O" will be part of an English word. Searching the lexicon is another talent. Even if we don't recognise every single character, it helps us guess words. Here I seek to classify individual words so the word can be converted into a digital form. Firstly, I took a dataset from Kaggle containing around 50000 handwritten words. Then I process and prepare the data for training and in the process, I remove the unreadable images. After processing the data and labels, I made a CRNN model which will use CNN and RNN sequentially. The model was then trained and checked for performance on the validation set. Finally, I took some custom inputs to check and the text was recognized with ease.

##  Model Architecture 
![HTR_Diagram](https://user-images.githubusercontent.com/40299522/148695687-c26d965f-804f-4fde-b2de-b4232dc03a5e.png)

##  Dataset ([Link](https://www.kaggle.com/landlord/handwriting-recognition))

This dataset consists of more than four hundred thousand handwritten names collected through charity projects. There are 206,799 first names and 207,024 surnames in total. The data was divided into a training set (331,059), testing set (41,382), and validation set (41,382) respectively.

##  Results
Correct characters predicted : 85.74%
Correct words predicted : 71.57%

## Limitation
There are multiple images in the training set which are not at all legible to the human eye. Removing such images will help in model's learning.

## Improvements
"*"More training samples should be used. This will improve the model's ability to learn and generalise. As Kaggle did not provide sufficient storage and computational resources only 30000 images are used for training. 
"*"The training set contains a number of images that are illegible to the human eye. Removing such images will improve the learning of the model.