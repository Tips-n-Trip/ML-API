import pandas as pd
from flask import Flask,request
import json

# Load and preprocess data
loadedModel = pd.read_hdf('./models/model.h5', 'df')
# Select relevant features for dataframe
data_model = loadedModel[['Place_Name','City','Images','Price','Rating','Lat', 'Long','Category','Description']]  # create a copy to avoid SettingWithCopyWarning
# Handle duplicates and missing data
data_model.drop_duplicates(inplace=True)
data_model.dropna(inplace=True)


app = Flask(__name__)

def recommend_places(city, budget, top_n=5): #top_n = wisata
    # Filter the data based on the input city and budget
    recommendations = data_model[(data_model['City'].str.contains(city, case=False, na=False)) & (data_model['Price'] <= budget)]
    
    # Calculate the absolute difference between the budget and the price, then sort the data based on this difference
    recommendations['abs_diff'] = abs(recommendations['Price'] - budget)
    recommendations.sort_values(by='abs_diff', ascending=True, inplace=True)

    # Return the top_n results & convert to dict
    recommended_place_list=recommendations.head(top_n).to_dict('records')

    # filter to only name
    attractions = []
    for place in recommended_place_list:
        attractions.append({'place_name' : place['Place_Name']})
        
    # convert ke json
    output = json.dumps({
        "attractions":attractions})
    
    return output


    

    return

@app.route("/")
def index():
    return {"message":"API ready to use!"}

@app.route("/generate",methods=['POST'])
def predict():
    data = request.json
    return recommend_places(data.get('destination'),data.get('budget'),data.get('duration')*3)

if __name__=="__main__":
    app.run(port=80)



