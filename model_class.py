import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
import time
import matplotlib.pyplot as plt


TEMP_MEAN = 0
TEMP_STD = 0

PRESS_MEAN = 0
PRESS_STD = 0
PRESS_DEFAULT = 1000

HUMIDITY_MEAN = 0
HUMIDITY_STD = 0
HUMIDITY_DEFAULT = 0.5

TIME_ZERO = pd.Timestamp('1970-01-01 00:00:00')
TIME_DELTA = '1h'

SEQ_LENGTH = 48
TO_PREDICT = 48


def preprocess_data(data, val_pct=0.2, to_predict=1):
    
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    
    pct = data.index[-(int(val_pct * len(data)))]
    
    print("pct:", pct, "data.index[0]:", data.index[0], "data.index[-1]:", data.index[-1], "len(data):", len(data))
    
    prev_days_x = deque(maxlen=SEQ_LENGTH)
    prev_days_y = deque(maxlen=to_predict)
    
    for index, row in zip(data.index, data.values):
        if index > data.index[-2 * to_predict]:
            break
        prev_days_x.append([])
        prev_days_y.append([])
        if to_predict == 1: 
            for n in range(len(row)):
                if (n < len(row) / 2):
                    if type(row[n]) is not tuple:
                        prev_days_x[len(prev_days_x) - 1].append(row[n])
    #                 else:
    #                     prev_days_x[len(prev_days_x) - 1].extend(row[n])
                else:
                    if type(row[n]) is not tuple:
                        prev_days_y[len(prev_days_y) - 1].append(row[n])
    #                 else:
    #                     prev_days_y[len(prev_days_y) - 1].extend(row[n])
                
            if len(prev_days_x) == SEQ_LENGTH:
    #             if (rand.rand() < val_pct) TODO! RANDOM SPLIT
                if index < pct:
                    train_x.append(np.array(prev_days_x))
                    train_y.append(np.array(prev_days_y[-1]))
                else:
                    val_x.append(np.array(prev_days_x))
                    val_y.append(np.array(prev_days_y[-1]))
                
        elif to_predict > 1:
            for n in range(len(row)):
                if (n < len(row) / 2):
                    if type(row[n]) is not tuple:
                        prev_days_x[len(prev_days_x) - 1].append(row[n])
                else:
                    if type(row[n]) is not tuple:
                        prev_days_y[len(prev_days_y) - 1].append(row[n])
                
            if len(prev_days_x) == SEQ_LENGTH:
                if index < pct:
                    train_x.append(np.array(prev_days_x))
                    train_y.append(np.array(prev_days_y))
                else:
                    val_x.append(np.array(prev_days_x))
                    val_y.append(np.array(prev_days_y))
    
    # shuffling the data
    rng_state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y)
        
    return (np.array(train_x), np.array(train_y)), (np.array(val_x), np.array(val_y))


def get_labels(data):
    """ returns the list of distinct labels in given data column """
    labels = list(set(data))
    return labels
    

def data_to_dicts(labels):
    """ returns pair of data to one-hot and one-hot to data dictionaries """
    data_to_oh = {x:tuple(1 if y == labels.index(x) else 0 
                    for y in range(len(labels))) 
                    for x in labels}
    
    oh_to_data = {y:x for x, y in data_to_oh.items()}
    
    return data_to_oh, oh_to_data


# ======= temp =======

def normalize_temp(temp):
    global TEMP_MEAN, TEMP_STD
    TEMP_MEAN = temp.mean()
    TEMP_STD = temp.std()
    return [(t - TEMP_MEAN) / TEMP_STD for t in temp]


def denormalize_temp(temp):
    return [t * TEMP_STD + TEMP_MEAN for t in temp]


def denormalize_temp_single(temp):
    return temp * TEMP_STD + TEMP_MEAN


# ======= press =======

def normalize_press(press):
    global PRESS_MEAN, PRESS_STD
    # 0's mess up mean and std calculations
    press = press.replace(0, PRESS_DEFAULT)
    PRESS_MEAN = press.mean()
    PRESS_STD = press.std()
    return [(p - PRESS_MEAN) / PRESS_STD for p in press]


def denormalize_press(press):
    return [p * PRESS_STD + PRESS_MEAN for p in press]


def denormalize_press_single(press):
    return press * PRESS_STD + PRESS_MEAN


# ======= hum =======

def normalize_humidity(hum):
    global HUMIDITY_MEAN, HUMIDITY_STD
    hum = hum.replace(0, HUMIDITY_DEFAULT)
    HUMIDITY_MEAN = hum.mean()
    HUMIDITY_STD = hum.std()
    return [(h - HUMIDITY_MEAN) / HUMIDITY_STD for h in hum]


def denormalize_humidity(hum):
    return [h * HUMIDITY_STD + HUMIDITY_MEAN for h in hum]


def denormalize_humidity_single(hum):
    return hum * HUMIDITY_STD + HUMIDITY_MEAN

# ===================

def normalize_time(times):
    """ converts date-time data column to a UNIX-style int (number of TIME_DELTA steps since TIME_ZERO) """
    times = [pd.Timestamp(time[:-6]) for time in times]
    times = [((time - TIME_ZERO) // pd.Timedelta(TIME_DELTA)) for time in times]
    return times


def one_hot_encode(data, data_to_oh):
    return [data_to_oh[d] for d in data]


def one_hot_decode(oh, oh_to_data):
    return [oh_to_data[o] for o in oh]



class LSTM_Model:

    def __init__(self, train_x, train_y, val_x, val_y, lstm_units=256, fc_units=64, lr=0.001):
        self.states = []

        self.x = tf.placeholder(tf.float32, [None, train_x.shape[1], train_x.shape[2]], name="x") # input for real data
        self.y = tf.placeholder(tf.float32, [None, train_x.shape[1], train_x.shape[2]], name="y") # output of a model

        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True, name="LSTM_cell")

        self.weights = tf.Variable(tf.random_normal([lstm_units, train_x.shape[1] * train_x.shape[2]]), name="fc_weights")
        self.biases = tf.Variable(tf.random_normal([train_x.shape[1] * train_x.shape[2]]), name="fc_biases")

        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

        self.lstm_units = lstm_units
        self.fc_units = fc_units
        self.lr = lr

        self.prediction = self.predict()
        self.cost = tf.reduce_mean(tf.abs(self.prediction - self.y))
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.losses = []


    def __del__(self):
        self.sess.close()


    def predict(self, x=None, y=None):
        # TOFIX
        if x is None:
            self.x = x
        if y != None:
            self.y = y
        x = tf.transpose(self.x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.train_x.shape[2]])
        x = tf.split(x, self.train_x.shape[1], 0)

        outputs, _ = tf.nn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(tf.nn.tanh(outputs[-1]), self.weights) + self.biases
        
        return tf.reshape(output, [-1, self.train_x.shape[1], self.train_x.shape[2]])


    def train(self, batch_size=256, epochs=15):
        for epoch in range(epochs):
            for i in range(len(self.train_x) // batch_size):
                epoch_x = self.train_x[i * batch_size:(i + 1) * batch_size]
                epoch_y = self.train_y[i * batch_size:(i + 1) * batch_size]

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                
                self.losses.append(c)

            c = self.sess.run(self.cost, feed_dict={self.x: self.val_x, self.y: self.val_y})
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', c)


    # def gan_train(self, lr=0.0001):
    #     pass




df = pd.read_csv("weatherHistory.csv", names = ['time', 'summary', 'precip', 'temp', 'app_temp', 'humidity', 'wind_speed', 'wind_bearing', 'visibility', 'loud_cover', 'pressure', 'daily_summary'], low_memory=False)

df = df.drop([0])
df = df.drop(['app_temp', 'wind_speed', 'wind_bearing', 'visibility', 'loud_cover', 'daily_summary'], axis=1) # TODO add wind_speed and other usefull data

df = df.drop(['summary', 'precip'], axis=1)

df.set_index('time', inplace=True)
df.index = normalize_time(df.index)

df = df.astype('float32')
print(df.columns.values)



df['temp'] = normalize_temp(df['temp'])
df['pressure'] = normalize_press(df['pressure'])
df['humidity'] = normalize_humidity(df['humidity'])
# df['humidity'] = df['humidity'].apply(pd.to_numeric)

print("temperature: mean={}, std={}".format(TEMP_MEAN, TEMP_STD))
print("pressure: mean={}, std={}".format(PRESS_MEAN, PRESS_STD))
print("humidity: mean={}, std={}".format(HUMIDITY_MEAN, HUMIDITY_STD))

df = df.sort_index()

for col in df.columns:
    df["future_{}".format(col)] = df["{}".format(col)].shift(-TO_PREDICT)


df = df.loc[:410500];
(train_x, train_y), (val_x, val_y) = preprocess_data(df, 0.3, TO_PREDICT)

model = LSTM_Model(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
model.train(epochs=3)
prediction = model.predict(x=val_x, y=val_y)
print(typeof(prediction))