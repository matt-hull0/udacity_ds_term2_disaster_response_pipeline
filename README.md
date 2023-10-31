# Disaster Response Pipeline 

This project has three components:

 1) An ETL pipeline that reads in messages and associated categories from disasters that can be used to train models.
    The data is loaded and cleaned ready for 2)

2) An ML pipeline that uses grid search to find the best parameters for a RandomForestClassifier, to predict
    the categories from the message

3) A web app that displays some key visualisations of the training data (see figure 1)
    as well as the ability to classify new messages (see figure 2)

### Figure 1 - App home screen showing visualtions on the training data
![](resources/home_screen.png)

### Figure 2 - Ability of app to classify new messages entered by the user
![](resources/msg_classification.png)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/
