import numpy as np 
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing


pd.set_option('display.float_format', lambda x: '%.5f' % x)
data=pd.read_csv("data1.csv",delimiter=';')
data=data.drop_duplicates()
data['PrevDay']=0.0
data['UpDay']=0
print('Creating UpDay')



for i in range(1,len(data.Close)-1):
    data.loc[i,'PrevDay']=data.Close[i-1]
    if(data.Close[i]<data.Close[i+1]):
        data.loc[i,'UpDay']=1
    else:
        data.loc[i,'UpDay']=0
print(data)

#Column Creation
data['mean']= (data.Close+data.High)/2





label= pd.DataFrame(data=data)
label = label['UpDay']

data = data[0:int(0.8*len(data.Close))]
label = label[0:8828]
data=data[['Close','High','Low','Open','Volume','PrevDay']]
test_data = data[int(0.8*len(data.Close)):]
test_label = label[int(0.8*len(data.Close)):]

print('Creating Data for ML...')

x_train = data.values
y_train = label.values

y_train = np.reshape(y_train,(-1,1))


x_test = test_data.values
y_test = test_label.values

y_test = np.reshape(y_test,(-1,1))


#pd.DataFrame(test_data).to_csv('Data/testdata.csv')
#pd.DataFrame(data).to_csv('Data/data.csv')
#pd.DataFrame(label).to_csv('Data/label.csv')
#pd.DataFrame(test_label).to_csv('Data/testlabel.csv')
print('Scaling Data for ML...')

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
print('Training....')
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)

predictions = model.predict(x_test)
binary_predictions = np.array([])
for i in range(0,len(predictions)):
    if(predictions[i][0]>predictions[i][1]):
        binary_predictions = np.append(binary_predictions,[0])
    else:
        binary_predictions = np.append(binary_predictions,[1])
        
binary_predictions = np.reshape(binary_predictions,(-1,1))       
print(binary_predictions)
import sys
np.set_printoptions(threshold=sys.maxsize)
print(y_test)
winning_trades = 0
for i in range(0,len(y_test)):
    if(binary_predictions[i]==y_test[i]):
            winning_trades+=1

winning_percentage = winning_trades/len(y_test)            
print('All Trades:',len(y_test))
print('Winning Trades:' ,winning_trades)
print('Losing Trades:' ,len(y_test)-winning_trades)
print('Winning Percentage:',round(winning_percentage,2),'%')