from model import make_model


model = make_model()
model.save('Model/complex_yolo.h5')
print('Done!')
