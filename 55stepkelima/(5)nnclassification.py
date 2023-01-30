# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
import pickle

hasil = 'hasilklasifikasinn.sav'
# load dataset
data = load('for5facesembeddings5november.npz')
trainX, trainy, valX, valy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'],data['arr_4'], data['arr_5']
print('Dataset: train=%d, val=%d, test=%d' % (trainX.shape[0], valX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
valX = in_encoder.transform(valX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
valy  = out_encoder.transform(valy)
testy = out_encoder.transform(testy)
# fit model
# model = SVC(kernel='linear', probability=True)
model = MLPClassifier()
model.fit(trainX, trainy)
#save model
filename = hasil
pickle.dump(model, open(filename, 'wb'))
# predict
yhat_train = model.predict(trainX)
yhat_val = model.predict(valX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_val = accuracy_score(valy, yhat_val)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, val=%.3f, test=%.3f' % (score_train*100, score_val*100, score_test*100))