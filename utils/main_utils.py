# Untuk pengolahan data
import json
import pandas as pd
import numpy as np

# Untuk pemodelan
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

#required class
class RecommenderNet(tf.keras.Model):
  def __init__(self, num_users, num_place, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_place = num_place
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.places_embedding = layers.Embedding(
        num_place,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.places_bias = layers.Embedding(num_place, 1)

    # Additional layers
    self.concat_layer = Concatenate()
    self.dense_1 = Dense(128, activation='relu')
    self.dropout_1 = Dropout(0.2)
    self.batch_norm_1 = BatchNormalization()
    self.dense_2 = Dense(64, activation='relu')
    self.dropout_2 = Dropout(0.2)
    self.batch_norm_2 = BatchNormalization()
    self.dense_output = Dense(1, activation='sigmoid')

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    places_vector = self.places_embedding(inputs[:, 1])
    places_bias = self.places_bias(inputs[:, 1])

    # Concatenate the embeddings 
    x = self.concat_layer([user_vector, places_vector])

    # Add one or more hidden layers
    x = self.dense_1(x)
    x = self.dropout_1(x)
    x = self.batch_norm_1(x)
    x = self.dense_2(x)
    x = self.dropout_2(x)
    x = self.batch_norm_2(x)

    # Output layer
    x = self.dense_output(x)
    return x
    #return tf.nn.sigmoid(x)

#fungsi encode
def dict_encoder(col, data):

  # Mengubah kolom suatu dataframe menjadi list tanpa nilai yang sama
  unique_val = data[col].unique().tolist()

  # Melakukan encoding value kolom suatu dataframe ke angka
  val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}

  # Melakukan proses encoding angka ke value dari kolom suatu dataframe
  val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
  return val_to_val_encoded, val_encoded_to_val

def run_model(destination, duration, budget,place,rating,model):

  # argumen 1, destination
  # mengubah data lokasi agar hanya dari kota tujuan
  place_s = place[place['City']==destination]

  # Merubah data rating agar hanya berisi rating pada tempat wisata dari Kota Spesifik
  rating_s = pd.merge(rating, place_s[['Place_Id']], how='right', on='Place_Id') #Jika implement kota spesifik => place ganti place_s

  # Merubah data user agar hanya berisi user yang pernah megunjungi wisata di Kota Spesifik
  ## user_s = pd.merge(user, rating_s[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')

  df = rating_s.copy()
  # Encoding User_Id
  user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id',df)

  # Mapping User_Id ke dataframe
  df['user'] = df['User_Id'].map(user_to_user_encoded)

  # Encoding Place_Id
  place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id',df)

  # Mapping Place_Id ke dataframe place
  df['place'] = df['Place_Id'].map(place_to_place_encoded)

  place_df = place[['Place_Id','Place_Name','Category','Price', 'City', 'Lat', 'Long', 'DorN', 'Description']]
  place_df.columns = ['id','place_name','category','price', 'city', 'lat', 'long', 'dorn', 'description']

  # argumen 2, user_id
  user_id = 1 #df.User_Id.sample(1).iloc[0]
  ## place_visited_by_user = df[df.User_Id == user_id] | NOTE: Not yet implemented, a lot of the code need to be removed for more efficiency
  place_visited_by_user = df[0:0]


  place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id'] 
  place_not_visited = list(
      set(place_not_visited)
      .intersection(set(place_to_place_encoded.keys()))
  )
  
  place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
  user_encoder = user_to_user_encoded.get(user_id)
  user_place_array = np.hstack(
      ([[user_encoder]] * len(place_not_visited), place_not_visited)
  )

  # parameter 3, duration
  # 1 hari berwisata => berkunjung ke 3 tempat

  hari_h = duration * 3

  # Mengambil data rekomendasi
  recommended_place_budget = 0

  loop_count = 0
  while (True):
    ratings = model.predict(user_place_array).flatten()

    top_ratings_indices = ratings.argsort()[-hari_h:][::-1]

    recommended_place_ids = [
        place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    ]
    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    total=recommended_place['price'].sum()
    loop_count=loop_count+1
    if(total<=budget):
      break
    if(loop_count>50):
      break


  # convert df to json
  recommended_place_list=recommended_place.to_dict('records')

  # filter to only name
  attractions = []
  for place in recommended_place_list:
    attractions.append({'place_name' : place['place_name']})


  # convert ke json
  output = json.dumps({
      "attractions":attractions})
  
  return output