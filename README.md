# Disaster Response Pipeline Project
This project uses Udacity processed data originally from <a href="https://www.figure-eight.com/dataset/combined-disaster-response-data/">Figure Eight's Disaster Response data</a> to train a classifier to classify new messages as one or more disaster related categories.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files
- /app/run.py - Python script to run the web app
- /data - Directory for the data files and python script for processing the data
    - process_data.py
    - disaster_messages.csv
    - disaster_categories.csv
- /models/train_classifier.py - Python script for training the classifier and saving the model

## Data
After removing duplicates the dataset contains 26,180 rows. The data contains messages from various disasters including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, and floods in Pakistan in 2010. The messages have been encoded with 36 different categories related to disaster response such as water, medical aid, and shelter. Messages can have more than one category and many of them do. This will require us to preform Multi-label Classification.

### Data Imbalance
The data is quite imbalanced. Some categories have very few examples (e.g. missing people has 301 observations). One category has no examples, "child alone".

<b>Techniques for dealing with imbalance:</b>
- Resampling: Because we are using a pipeline and classification is multi-label, resampling functions (such as RandomOverSampling or SMOTE) could not be used. Testing with just one of the more imbalanced categories showed that it would be helpful.

- Scoring: 
	- With an imbalanced dataset, accuracy is not a good performance measure because it's only reflecting the underlying class distribution. Recall or F1 score are better choices. For this project, recall was used because false negatives (e.g. missing a message that should be classified as disaster related) is a bigger problem than false positives.
	- Hamming Loss is a useful metric in multi-label classification.  It is the fraction of labels that are incorrectly predicted.

- Model Selection: Some machine learning algorithms handle class imbalance better than others, e.g. Random Forest and Boosted learners. AdaBoost was found to be the better model for this dataset.

## Results
Results including "child alone":
AdaBoostCV, Recall: 0.352, Hamming Loss: 0.0532

Results with "child alone" removed:

| Model | Recall | Hamming Loss |
| --- | --- | --- |
| RandomForest| 0.566 | 0.0592|
| RandomForestCV| 0.588 | 0.0553 |
| AdaBoost| 0.651 | 0.0533 |
| AdaBoostCV| 0.653 | 0.0534 |

![Chart: Recall](/images/recall.png)
