import pandas as pd
import pickle
from flask import Flask,request
from utils.main_utils import RecommenderNet, run_model

# load datasets
rating = pd.read_csv('./datasets/dataset_rating.csv')
place = pd.read_csv('./datasets/dataset_lokasi.csv')
# user = pd.read_csv('dataset_user.csv')
# load model
model = pickle.load(open('./models/model.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return {"message":"API ready to use!"}

    # you can add your own routes here as needed
@app.route("/generate",methods=['POST'])
def generate():
    data = request.json
    return run_model(data.get('destination'),data.get('duration'),data.get('budget'),place,rating,model)


if __name__=="__main__":
    app.run(port=8080)
