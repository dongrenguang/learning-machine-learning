import tensorflow as tf
import tensorflow_datasets as tfds

ds, ds_info = tfds.load("movielens/100k-ratings", split='train', shuffle_files=True, with_info=True)
df = tfds.as_dataframe(ds, ds_info)
features = ['bucketized_user_age', 'movie_genres', 'movie_id', 'user_gender', 'user_id', 'user_occupation_label', 'user_zip_code']
def ds_preprocess(x):
    label = tf.cast(x['user_rating'] >= 4.0, tf.int64)
    return {key: x[key] for key in features}, label
ds = ds.map(ds_preprocess).shuffle(len(ds))

train_data_size = int(0.8 * len(ds))
ds_train = ds.take(train_data_size).ragged_batch(batch_size=1000)
ds_test = ds.skip(train_data_size).ragged_batch(batch_size=1000)

bucketized_user_age_lookup = tf.keras.layers.IntegerLookup(vocabulary=df['bucketized_user_age'].unique().astype(int))
movie_genres_lookup = tf.keras.layers.IntegerLookup(vocabulary=df['movie_genres'].explode().unique().astype(int))
movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=df['movie_id'].unique())
user_gender_lookup = tf.keras.layers.IntegerLookup(vocabulary=df['user_gender'].unique().astype(int))
user_id_lookup = tf.keras.layers.StringLookup(vocabulary=df['user_id'].unique())
user_occupation_label_lookup = tf.keras.layers.IntegerLookup(vocabulary=df['user_occupation_label'].unique())
user_zip_code_lookup = tf.keras.layers.StringLookup(vocabulary=df['user_zip_code'].unique())

bucketized_user_age_embed = tf.keras.layers.Embedding(bucketized_user_age_lookup.vocabulary_size(), 4)
movie_genres_embed = tf.keras.layers.Embedding(movie_genres_lookup.vocabulary_size(), 4)
movie_id_embed = tf.keras.layers.Embedding(movie_id_lookup.vocabulary_size(), 16)
user_gender_embed = tf.keras.layers.Embedding(user_gender_lookup.vocabulary_size(), 4)
user_id_embed = tf.keras.layers.Embedding(user_id_lookup.vocabulary_size(), 16)
user_occupation_label_embed = tf.keras.layers.Embedding(user_occupation_label_lookup.vocabulary_size(), 4)
user_zip_code_embed = tf.keras.layers.Embedding(user_zip_code_lookup.vocabulary_size(), 16)

inputs = {}
inputs['bucketized_user_age'] = tf.keras.Input(shape=(), name='bucketized_user_age', dtype=tf.float32)
inputs['movie_genres'] = tf.keras.Input(shape=(None,), name='movie_genres', dtype=tf.int64)
inputs['movie_id'] = tf.keras.Input(shape=(), name='movie_id', dtype=tf.string)
inputs['user_gender'] = tf.keras.Input(shape=(), name='user_gender', dtype=tf.bool)
inputs['user_id'] = tf.keras.Input(shape=(), name='user_id', dtype=tf.string)
inputs['user_occupation_label'] = tf.keras.Input(shape=(), name='user_occupation_label', dtype=tf.int64)
inputs['user_zip_code'] = tf.keras.Input(shape=(), name='user_zip_code', dtype=tf.string)

preprocessed = [
    bucketized_user_age_embed(bucketized_user_age_lookup(inputs['bucketized_user_age'])),
    tf.reduce_mean(movie_genres_embed(movie_genres_lookup(inputs['movie_genres'])), axis=1, keepdims=False),
    movie_id_embed(movie_id_lookup(inputs['movie_id'])),
    user_gender_embed(user_gender_lookup(inputs['user_gender'])),
    user_id_embed(user_id_lookup(inputs['user_id'])),
    user_occupation_label_embed(user_occupation_label_lookup(inputs['user_occupation_label'])),
    user_zip_code_embed(user_zip_code_lookup(inputs['user_zip_code']))
]
preprocessed_concat = tf.keras.layers.Concatenate()(preprocessed)
preprocessor = tf.keras.Model(inputs, preprocessed_concat)

body = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
x = preprocessor(inputs)
result = body(x)
model = tf.keras.Model(inputs, result)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', 'AUC'])
model.fit(ds_train, epochs=10, validation_data=ds_test,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
