# Disaster Response Pipeline Project

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/



#### 1, Libraries used
Regular expression - re
Process dataframe - numpy, pandas
Database access - sqlalchemy
Natural Language Processing - nltk
Machine Learning - sklearn
Save model - pickle
System access - sys


#### 2. Project Motivation 

​Practice building piplines as a data engineering. This project includes ETL pipline, NLP pipline and Machine Learning pipline. 
The topic of this project is really interesting and meaningful. The data coming from tweets and text messages from real life disaster and processed by Figure Eight. When it comes to disaster, millions of messages coming in. By compeleting this project, we can build a supervised learning model to filter out messages that are really important and need attention by disaster response professional. Then different organizations and take care out each category of diasters. 


#### 3. Technical details 

1. Build ETL pipline to extract and clean the data, then saved into database
2. Build Machine Learning pipline with NLP pipline to process text information and build model to categorize messages
3. Use Flask + Plotly to show Disaster Response Dashboard


#### 4. Files in Repository
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs web app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # ETL process - read in data and clean, then load to database
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py # NLP + Machine Learning Pipeline
|- classifier.pkl # saved model
README.md



#### 5.Licensing, Authors, Acknowledgements 

Data and pipline template coming from Udacity Data Science Nano Program

