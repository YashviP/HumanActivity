import pickle
import numpy as np
from sklearn.svm import LinearSVC


with open('feature_mat.pkl','rb') as f:
     feature_mat = pickle.load(f)

with open('label.pkl','rb') as f:
     labels = pickle.load(f)

print(feature_mat)
print(labels)

labels=labels.ravel()
labels=np.array(labels,dtype=int)
print(labels)
print(labels.shape)

model=LinearSVC(max_iter=3000)
model.fit(feature_mat,labels)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

