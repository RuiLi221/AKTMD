import numpy as np 
import tensorflow as tf 
from tfkan import layers
from tfkan.layers import DenseKAN, Conv1DKAN

X = np.load("/***")
Y = np.load("/***")

train_x = X[:236911]
val_x = X[236911:]
train_y = Y[:236911]
val_y = Y[236911:]
W=500

lacc_x = tf.keras.layers.Input(shape=(W, 1))
lacc_y = tf.keras.layers.Input(shape=(W, 1))
lacc_z = tf.keras.layers.Input(shape=(W, 1))

gyr_x = tf.keras.layers.Input(shape=(W, 1))
gyr_y = tf.keras.layers.Input(shape=(W, 1))
gyr_z = tf.keras.layers.Input(shape=(W, 1))

mag_x = tf.keras.layers.Input(shape=(W, 1))
mag_y = tf.keras.layers.Input(shape=(W, 1))
mag_z = tf.keras.layers.Input(shape=(W, 1))

pressure = tf.keras.layers.Input(shape=(W, 1))

dropout_rate1 = 0.1
def Conv1DKAN_block(X, Filters, kernel_sizes):
    F1, F2, F3, F4 = Filters
    K1, K2, K3, K4 = kernel_sizes
    X = Conv1DKAN(F1, K1, padding='same')(X)
    X = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(X)
    X = tf.keras.layers.Dropout(dropout_rate1)(X)
    return X

def ConvKAN_net(X_input):
    X = Conv1DKAN_block(X_input, Filters=[6, 6, 6, 6], kernel_sizes=[3, 3, 3, 3])
    return X

def ConvKAN_layer(lacc_x, lacc_y, lacc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, pressure):
    lacc_x1 = ConvKAN_net(lacc_x)
    lacc_y1 = ConvKAN_net(lacc_y)
    lacc_z1 = ConvKAN_net(lacc_z)

    gyr_x1 = ConvKAN_net(gyr_x)
    gyr_y1 = ConvKAN_net(gyr_y)
    gyr_z1 = ConvKAN_net(gyr_z)

    mag_x1 = ConvKAN_net(mag_x)
    mag_y1 = ConvKAN_net(mag_y)
    mag_z1 = ConvKAN_net(mag_z)

    pressure1 = ConvKAN_net(pressure)

    cha_KAN_lacc = tf.keras.layers.concatenate([lacc_x1, lacc_y1, lacc_z1])
    cha_KAN_gyr = tf.keras.layers.concatenate([gyr_x1, gyr_y1, gyr_z1])
    cha_KAN_mag = tf.keras.layers.concatenate([mag_x1, mag_y1, mag_z1])
    cha_KAN_pre =  tf.keras.layers.concatenate([pressure1])

    return cha_KAN_lacc, cha_KAN_gyr, cha_KAN_mag, cha_KAN_pre
    
f = 6
k = 3
dropout_rate2 = 0.1
def linked_KANlayer(cha_KAN_lacc, cha_KAN_gyr, cha_KAN_mag, cha_KAN_pre):
    linked_KAN_lacc = Conv1DKAN(f, k, padding='same')(cha_KAN_lacc)
    linked_KAN_lacc1 = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(linked_KAN_lacc)
    linked_KAN_lacc1 = tf.keras.layers.Dropout(dropout_rate2)(linked_KAN_lacc1)

    linked_KAN_gyr = Conv1DKAN(f, k, padding='same')(cha_KAN_gyr)
    linked_KAN_gyr1 = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(linked_KAN_gyr)
    linked_KAN_gyr1 = tf.keras.layers.Dropout(dropout_rate2)(linked_KAN_gyr1)
    
    linked_KAN_mag = Conv1DKAN(f, k, padding='same')(cha_KAN_mag)
    linked_KAN_mag1 = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(linked_KAN_mag)
    linked_KAN_mag1 = tf.keras.layers.Dropout(dropout_rate2)(linked_KAN_mag1)

    linked_KAN_pre = Conv1DKAN(f, k, padding='same')(cha_KAN_pre)    
    linked_KAN_pre1 = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(linked_KAN_pre)
    linked_KAN_pre1 = tf.keras.layers.Dropout(dropout_rate2)(linked_KAN_pre1)

    linked_KAN = tf.keras.layers.concatenate([linked_KAN_lacc1, linked_KAN_gyr1, linked_KAN_mag1, linked_KAN_pre1])
    linked_KAN = tf.keras.layers.Dropout(dropout_rate2)(linked_KAN)
    return linked_KAN

def MultiHeadAttention_layer(x, y):
    att = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=128, output_shape=128, dropout=0.1)(x, y)
    att1 = tf.keras.layers.AveragePooling1D(32)(att)
    att2 = tf.keras.layers.Flatten()(att1)
    return att2

def mlp_layer(x):
    x = DenseKAN(128, grid_size = 5)(x)
    x = DenseKAN(8, grid_size = 5)(x)
    output = tf.keras.layers.Activation('softmax')(x)
    return output

cha_KAN_lacc, cha_KAN_gyr, cha_KAN_mag, cha_KAN_pre = ConvKAN_layer(
    gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure)
all_resnet = linked_KANlayer(cha_KAN_lacc, cha_KAN_gyr, cha_KAN_mag, cha_KAN_pre)
att_new = MultiHeadAttention_layer(all_resnet, all_resnet)
output = mlp_layer(att_new)

model = tf.keras.Model(inputs=[
    gyr_x, gyr_y, gyr_z,
    lacc_x, lacc_y, lacc_z,
    mag_x, mag_y, mag_z, pressure
],outputs=output)

model.summary()

initial_learning_rate = 1e-3
my_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=my_optimizer,  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_SHL.h5', 
    monitor='sparse_categorical_accuracy', 
    save_best_only=True, 
    mode='max'
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='sparse_categorical_accuracy', 
    patience=15, 
    mode='max', 
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='sparse_categorical_accuracy', 
    factor=0.2,        
    patience=5,          
    min_lr=1e-3,       
    verbose=1           
)

model.fit(
    [train_x[:, :, 0], train_x[:, :, 1], train_x[:, :, 2], train_x[:, :, 3],train_x[:, :, 4],
     train_x[:, :, 5], train_x[:, :, 6], train_x[:, :, 7], train_x[:, :, 8], train_x[:, :, 9]],
    train_y, validation_data=([val_x[:, :, 0], val_x[:, :, 1], val_x[:, :, 2], val_x[:, :, 3],
    val_x[:, :, 4], val_x[:, :, 5], val_x[:, :, 6], val_x[:, :, 7], val_x[:, :, 8], val_x[:, :, 9]],
    val_y), epochs = 200, shuffle= True , batch_size = 512, callbacks = [checkpoint_callback, early_stopping_callback, reduce_lr])

model.load_weights('best_model_SHL.h5')

predictions = model.predict(
    [val_x[:, :, 0], val_x[:, :, 1], val_x[:, :, 2], val_x[:, :, 3],
     val_x[:, :, 4], val_x[:, :, 5], val_x[:, :, 6], val_x[:, :, 7],
     val_x[:, :, 8], val_x[:, :, 9]]
)

predictions = [np.argmax(p) for p in predictions]
lv = np.reshape(train_y, newshape=-1)
accuracy = 0
cnf = [[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]]
for (p, t) in zip(predictions, lv):
  p = int(p)
  t = int(t)
  cnf[t][p] += 1
  if p == t:
    accuracy += 1
accuracy /= float(len(predictions))
print('acc', accuracy)
print(np.array(cnf))
print('1: %f\n2: %f\n3: %f\n4: %f\n5: %f\n6: %f\n7: %f\n8: %f\n' % (cnf[0][0] / float(np.sum(cnf[0])),
                                                                    cnf[1][1] / float(np.sum(cnf[1])),
                                                                    cnf[2][2] / float(np.sum(cnf[2])),
                                                                    cnf[3][3] / float(np.sum(cnf[3])),
                                                                    cnf[4][4] / float(np.sum(cnf[4])),
                                                                  cnf[5][5] / float(np.sum(cnf[5])),
                                                                  cnf[6][6] / float(np.sum(cnf[6])),
                                                                  cnf[7][7] / float(np.sum(cnf[7])),
                                                                    ))



