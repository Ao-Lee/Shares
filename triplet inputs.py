import numpy as np
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, GlobalMaxPooling2D, Dropout, Lambda
from keras.applications.resnet50 import ResNet50, preprocess_input

############## Settings ###########################
batch_size = 8
image_size = 224
embedding_dim = 5

############## DATA ###########################
def GetRandomImage():
    return np.random.randint(low=0, high=256, size=[image_size,image_size,3])
    
def GetTriplet():
    a = GetRandomImage()
    b = GetRandomImage()
    c = GetRandomImage()
    return a,b,c

def Generate():
    while True:
        list_a = []
        list_b = []
        list_c = []

        for i in range(batch_size):
            a, b, c = GetTriplet()
            list_a.append(a)
            list_b.append(b)
            list_c.append(c)
            
        A = preprocess_input(np.array(list_a, dtype='float32'))
        B = preprocess_input(np.array(list_b, dtype='float32'))
        C = preprocess_input(np.array(list_c, dtype='float32'))
        label = np.ones(batch_size)
        yield [A, B, C], label


train_generator = Generate()
test_generator = Generate()
batch = next(train_generator)
       
############## LOSS ########################### 
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

A = np.ones([batch_size,embedding_dim])
A /= embedding_dim
A = K.variable(A)
    
def fake_loss(X):
    item, _,_ = X
    # item = X
    loss = K.sum(A * item, axis=-1, keepdims=True)
    return loss
    
############## Model ########################### 
def GetBaseModel():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    dense_1 = Dense(embedding_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model

def GetMyModel(base_model):
    input_1 = Input((image_size,image_size,3))
    input_2 = Input((image_size,image_size,3))
    input_3 = Input((image_size,image_size,3))

    r1 = base_model(input_1)
    r2 = base_model(input_2)
    r3= base_model(input_3)

    loss = Lambda(fake_loss)([r1,r2,r3]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000003))
    # print(model.summary())
    return model



base_model = GetBaseModel()
model = GetMyModel(base_model)

print(model.predict_on_batch(batch[0]))


model.fit_generator(train_generator, 
                    validation_data=test_generator, 
                    epochs=5, 
                    verbose=2, 
                    workers=4,
                    steps_per_epoch=20, 
                    validation_steps=30)


print(model.predict_on_batch(batch[0]))