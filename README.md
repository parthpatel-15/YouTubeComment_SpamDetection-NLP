# YouTubeComment_SpamDetection-NLP
The code demonstrates a complete pipeline for spam detection in YouTube comments using text vectorization, TF-IDF, and a Multinomial Naive Bayes classifier. It also evaluates the model accuracy through cross-validation and tests it on new comments.

# Data Loading and Exploration:
Loads the dataset and explores its shape, first few records, data types of columns, and checks for null values.
# Data Pre-processing:
Drops unnecessary columns ('COMMENT_ID', 'AUTHOR', 'DATE').
Shuffles the dataset using pandas.sample with frac=1.
# Text Vectorization:
Utilizes CountVectorizer to convert text comments into numerical data.
Applies lowercase and removes English stopwords.
Computes the count matrix.
# Term Frequency times Inverse Document Frequency (TF-IDF):
Creates a TF-IDF transformer.
Computes TF-IDF values for the count matrix.
# Model Training:
Splits the dataset into 75% for training and 25% for testing.
Uses Multinomial Naive Bayes classifier to fit the training data.
# Cross-validation:
Performs 5-fold cross-validation on the training data.
Prints the mean accuracy of the model.
# Testing the Model:
Tests the model on the test data.
Prints the confusion matrix and accuracy of the model.
# Predicting New Comments:
Generates 6 new comments (4 non-spam and 2 spam).
Passes the comments through the classifier and prints the predicted results.

# output :
<img width="500" alt="Screenshot 2023-11-15 at 4 19 25 PM" src="https://github.com/parthpatel-15/YouTubeComment_SpamDetection-NLP/assets/79576096/460ee9af-b184-4f9b-9509-44bd4feb0de4">



