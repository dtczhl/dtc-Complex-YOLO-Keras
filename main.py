from model import make_model
import numpy as np
import os

model = make_model()

x = np.random.random((1, 1024, 512, 3))
y = np.random.random((1, 32, 16, 75))

model.compile('sgd', 'mse')  # TODO: actual loss implementation

model.fit(x, y)

# save trained model to Model folder
os.mkdir('Model')
model.save('Model/complex_yolo.h5')

print('Done!')
