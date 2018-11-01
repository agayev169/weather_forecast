#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import deque


# In[2]:


# constants used in pre-processing

TEMP_COEF = 50

PRESS_SHIFT = 1000
PRESS_COEF = 100
PRESS_DEFAULT = 1000

TIME_ZERO = pd.Timestamp('1970-01-01 00:00:00')
TIME_DELTA = '1h'

SEQ_LENGTH = 48
PERIOD_TO_PREDICT = 1


# In[3]:


# functions for cleaning the data

def preprocess_data(data, val_pct=0.2):
   
   train_x = []
   train_y = []
   val_x = []
   val_y = []
   
   pct = data.index[-(int(val_pct * len(data)))]
   
   print("pct:", pct, "data.index[0]:", data.index[0], "data.index[-1]:", data.index[-1], "len(data):", len(data))
   
   prev_days_x = deque(maxlen=SEQ_LENGTH)
   prev_days_y = deque(maxlen=SEQ_LENGTH)
   
   for index, row in zip(data.index, data.values):
       if index > data.index[-2*PERIOD_TO_PREDICT]:
           break
       prev_days_x.append([])
       prev_days_y.append([])
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


def normalize_temp(temp):
   return [float(t) / TEMP_COEF for t in temp]


def denormalize_temp(temp):
   return [t * TEMP_COEF for t in temp]


def normalize_press(press):
   press = [float(p) for p in press]
   for i in range(len(press)):
       if press[i] == 0:
           press[i] = press[i-1] if i != 0 else PRESS_DEFAULT

   return [(p - PRESS_SHIFT) / PRESS_COEF for p in press]


def denormalize_press(press):
   return [p * PRESS_COEF + PRESS_SHIFT for p in press]


def normalize_time(times):
   """ converts date-time data column to a UNIX-style int (number of TIME_DELTA steps since TIME_ZERO) """
   times = [pd.Timestamp(time[:-6]) for time in times]
   times = [((time - TIME_ZERO) // pd.Timedelta(TIME_DELTA)) for time in times]
   return times


# def denormalize_time(time):
# TODO


def one_hot_encode(data, data_to_oh):
   return [data_to_oh[d] for d in data]


def one_hot_decode(oh, oh_to_data):
   return [oh_to_data[o] for o in oh]


# In[4]:


df = pd.read_csv("weatherHistory.csv", names = ['time', 'summary', 'precip', 'temp', 'app_temp', 'humidity', 'wind_speed', 'wind_bearing', 'visibility', 'loud_cover', 'pressure', 'daily_summary'], low_memory=False)

df = df.drop([0])
df = df.drop(['app_temp', 'wind_speed', 'wind_bearing', 'visibility', 'loud_cover', 'daily_summary'], axis=1) # TODO add wind_speed and other usefull data

df.set_index('time', inplace=True)
df.index = normalize_time(df.index)

df.head()
print(df.columns.values)


# In[5]:


summary_labels = get_labels(df['summary'])
# print("len(summary_labels):", len(summary_labels))

# our training data contains nans when there is no precipitation
df['precip'] = df['precip'].fillna("clear")
precip_labels = get_labels(df['precip'])
# print("len(precip_labels):", len(precip_labels))

# daily_summary_labels = get_labels(df['daily_summary'])
# print("len(daily_summary_labels):", len(daily_summary_labels))


summary_to_oh, oh_to_summary = data_to_dicts(summary_labels)
precip_to_oh, oh_to_precip = data_to_dicts(precip_labels)

# print(summary_to_oh, oh_to_summary, sep='\n\n')
# print(precip_to_oh, oh_to_precip, sep='\n\n')

df['summary'] = one_hot_encode(df['summary'], summary_to_oh)
# df['summary'].head()
df['precip'] = one_hot_encode(df['precip'], precip_to_oh)
# df['precip'].head()


# In[6]:


df['temp'] = normalize_temp(df['temp'])
df['pressure'] = normalize_press(df['pressure'])
df['humidity'] = df['humidity'].apply(pd.to_numeric)

# print(denormalize_temp(df['temp'])[:5])
# print(denormalize_press(df['pressure'])[:5])
# print(min(df['temp']), max(df['temp']), '\n', min(df['pressure']), max(df['pressure']))


# In[7]:


# sorting data by index
df = df.sort_index()


# In[8]:


# we shift values so that each row has a corresponding future row
for col in df.columns:
    df["future_{}".format(col)] = df["{}".format(col)].shift(-PERIOD_TO_PREDICT)


# In[9]:


(train_x, train_y), (val_x, val_y) = preprocess_data(df, 0.3)

print("length of train x:", len(train_x))
print("length of train y:", len(train_y))
print("length of val x:", len(val_x))
print("length of val y:", len(val_y))
print("ratio:", len(val_x) / (len(train_x) + len(val_x)))

# # Model

# In[10]:


import tensorflow as tf


# In[11]:


print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)


# In[12]:


# constants used in the model

LSTM_LAYERS = 1
LSTM_UNITS = 128

FC_LAYERS = 1
FC_UNITS = 128

# INPUT_DIM = (len(summary_labels) + len(precip_labels) + 3) * SEQ_LENGTH
INPUT_DIM = 3 * SEQ_LENGTH
OUTPUT_DIM = 3 * SEQ_LENGTH

num_features = 3
batch_size = 128


# In[13]:


hm_epochs = 1
rnn_size = 128

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, num_features])
y = tf.placeholder(tf.float32, [None, num_features])

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, num_features])),
             'biases':tf.Variable(tf.random_normal([num_features]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, num_features])
    x = tf.split(x, SEQ_LENGTH, 0)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
#             for _ in range(int(mnist.train.num_examples/batch_size)):
            for i in range(int(len(train_x)/batch_size)):
#                 epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = train_x[i * batch_size:(i + 1) * batch_size]
                epoch_y = train_y[i * batch_size:(i + 1) * batch_size]
#                 epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of', hm_epochs,'loss:', epoch_loss)

#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy, _ = tf.metrics.mean_squared_error(labels=y, predictions=prediction)
#         print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))
#         print('Accuracy:',accuracy.eval({x:val_x, y:val_y}))
        print('Accuracy:', sess.run(accuracy, feed_dict={x: val_x, y: val_y}))

train_neural_network(x)


# In[ ]:




