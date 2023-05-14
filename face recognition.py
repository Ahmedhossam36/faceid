import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import uuid


# making new dirictories for the files and data
# pos_path = os.path.join('/home/ahmed/Downloads/gym_cityflow/faceid','data','positive')
# neg_path = os.path.join('/home/ahmed/Downloads/gym_cityflow/faceid','data','negative')
# anc_path = os.path.join('/home/ahmed/Downloads/gym_cityflow/faceid','data','anchor')

# os.makedirs(pos_path)
# os.makedirs(neg_path)
# os.makedirs(anc_path)

# extracting the images in the dataset in the negative folder

# for directory in os.listdir('/home/ahmed/Downloads/gym_cityflow/faceid/lfw'):
#     for file in os.listdir(os.path.join('/home/ahmed/Downloads/gym_cityflow/faceid/lfw',directory)):
#         ex_path=os.path.join('/home/ahmed/Downloads/gym_cityflow/faceid/lfw',directory,file)
#         new_path = os.path.join(neg_path,file)
#         os.replace(ex_path,new_path)

anc_path = "/home/ahmed/Downloads/faceid/data/anchor"
pos_path = "/home/ahmed/Downloads/faceid/data/positive"
neg_path = "/home/ahmed/Downloads/faceid/data/negative"
# cap = cv2.VideoCapture(0)
# while  cap.isOpened():
#     ret , frame = cap.read()
#     frame = frame[120:120+250,200:200+250,:]
#     cv2.imshow('image',frame)
#     if cv2.waitKey(20) & 0XFF == ord('q'):
#         break
#     if cv2.waitKey(20) & 0XFF == ord('a'):
#         imgname = os.path.join(anc_path,"{}.jpg".format(uuid.uuid1()))
#         cv2.imwrite(imgname,frame)
#     if cv2.waitKey(20) & 0XFF == ord('p'):
#         imgname = os.path.join(pos_path,"{}.jpg".format(uuid.uuid1()))
#         cv2.imwrite(imgname,frame)

##########################

anchor = tf.data.Dataset.list_files(anc_path + '/*.jpg').take(1000)
postive = tf.data.Dataset.list_files(pos_path + '/*.jpg').take(1000)
negative = tf.data.Dataset.list_files(neg_path + '/*.jpg').take(1000)  # type: ignore


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255
    return img


dir = anchor.as_numpy_iterator()
img = preprocess(dir.next())
# plt.imshow(img)
# plt.show()

postives = tf.data.Dataset.zip((anchor, postive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = postives.concatenate(negatives)


# samples = data.as_numpy_iterator()
# example = samples.next()
# print(example)

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

train_data = data.take(round(len(data) * 0.7))  # take 70% of the data
train_data = train_data.batch(16)  # make them in groups of 16
train_data = train_data.prefetch(8)  # start preprocessing the next 8 pics for acceleration

test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_img')

    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()


# embedding.summary()

class l1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


l1 = l1Dist()


def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_image', shape=(100, 100, 3))
    siamese_layer = l1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

checkpoints_dir = '/home/ahmed/Downloads/faceid/checkpoints'
checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # get anchor and positive/negative image
        x = batch[:2]
        # get label
        y = batch[2]

        # forward pass
        yhat = siamese_model(x, training=True)

        loss = binary_cross_loss(y, yhat)
    print(loss)
    # calc_gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # calc update weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, epochs):
    for epoch in range(1, epochs + 1):
        print('\n epoch{}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx + 1)

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoints_prefix)


epochs = 50
# train(train_data, epochs)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()
test_var = test_data.as_numpy_iterator().next()

# make predictions

y_hat = siamese_model.predict([test_input, test_val])

[1 if prediction > 0.5 else 0 for prediction in y_hat]

# creating a metric object
m = Recall()
# calc the recall value
m.update_state(y_true, y_hat)

m.result().numpy()
print(m)

# creating a metric object
m = Precision()
# calc the recall value
m.update_state(y_true, y_hat)

m.result().numpy()
print(m)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(test_input[0])
plt.subplot(1, 2, 2)
plt.imshow(test_val[0])
plt.show()

# saving the model

siamese_model.save("./siamesemodel.h5")
model = tf.keras.models.load_model("./siamesemodel.h5", custom_objects={'l1Dist': l1Dist,
                                                                        'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
model.predict([test_input, test_val])


# resl_time

# verification function
def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('real_time_data', 'verification_img')):
        input_img = preprocess(os.path.join('real_time_data', 'input_img', 'input_img.jpg'))
        validation_img = preprocess(os.path.join('real_time_data', 'verification_img', image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.path.join('real_time_data', 'verification_img'))
    verified = verification > verification_threshold

    return results, verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]
    cv2.imshow('verification', frame)

    # verification trigger
    if cv2.waitKey(20) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join('real_time_data', 'input_img', 'input_img.jpg'), frame)
        results, verified = verify(model, 0.5, 0.5)
        print(verified)
        print(results)

    if cv2.waitKey(20) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllwindows()
