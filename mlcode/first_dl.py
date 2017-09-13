#coding:utf-8
#第一次尝试用keras 建深度学习
#在kuki/.keras/keras.json 里修改backen 改为theano
#还没用数据尝试  先放在这……
from keras.models import Sequential
model = Sequential
from keras.layers import Dense, Activation
model.add(Dense(units=64,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=32)

model.train_on_batch(x_batch,y_batch)

loss_and_metrics = model.evaluate(x_test,y_test,batch_size=128)

classes = model.predict(x_test,batch_size=128)