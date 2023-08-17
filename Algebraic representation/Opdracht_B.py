"""
MNIST opdracht B: "Conv Dense"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je alleen nog als laatste layer een dense layer.

De opdracht bestaat uit drie delen: B1 tm B3.

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
Zorg ervoor dat je de onderstaande code volledig begrijpt.

Tip: stap in de Debugger door de code, en bestudeer de tussenresultaten.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MavTools_NN import ViewTools_NN

#tf.random.set_seed(0) #for reproducability

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])
print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.show()
plt.grid(False)

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# This time, we don't flatten the images, because that would destroy the 
# locality benefit of the convolution layers.

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Opdracht B1: 
    
Voeg als hidden layers ALLEEN Convolution en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,97.

Voorbeelden van layers:
    layers.Dropout(getal)
    layers.Conv2D(getal, kernel_size=(getal, getal), padding="valid" of "same")
    layers.Flatten()
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
je ervaringen daarbij en probeer die ervaringen te verklaren.

model:
- Conv2D layer met een kernel_size=(3, 3) met 4 filters
- MaxPooling2D met pool_size=(2, 2)
- Dropout van 0.3
- flatten
- de dense layer
Test loss: 0.0103
Test accuracy: 0.9501


model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout( 0.3 ),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            
            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
Test loss: 0.0077
Test accuracy: 0.9604
(pas na Epoch 500, bij epoch 256: acc 0.9530)

--------------------------------------------------------------------
dropout van 0.3
na 150 epochs: accuracy 0.9266
dropout van 0.1
  na 150 epochs: accuracy 0.9265

---------------------------------------------------------------------
eerste conv layer is nu 5*5
extra maxpooling voor flatten
  na 120 epochs: accuracy 0.8594

-----------------------------------
maxpooling weggehaald
  Test accuracy: 0.9444

--------------------------------------
extra Conv2D 3*3 na de 5*5
  Test accuracy: 0.9493

---------------------------------------
5*5 terug veranderd in 3*3

model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout( 0.3 ),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            
            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )

Test accuracy: 0.9567

----------------------------------------------------------------------------
eerste conv filter veranderd van 4 filters naar 8 filters
  Test accuracy: 0.9653

--------------------------------------------------------------------
2e conv filter ook veranderd van 4 filters naar 8 filters
  Test accuracy: 0.9692

--------------------------------------------------------------------------
extra conv2d filter toegevoegd met 4 filters
model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(8, kernel_size=(3, 3), padding="valid"),
            keras.layers.Conv2D(8, kernel_size=(3, 3), padding="valid"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout( 0.3 ),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),

            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )

Test accuracy: 0.9713



Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_B.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(8, kernel_size=(3, 3), padding="valid"),
            keras.layers.Conv2D(8, kernel_size=(3, 3), padding="valid"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout( 0.3 ),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),

            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht B2:
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 8)         80

 conv2d_1 (Conv2D)           (None, 24, 24, 8)         584

 max_pooling2d (MaxPooling2D  (None, 12, 12, 8)        0
 )

 dropout (Dropout)           (None, 12, 12, 8)         0

 conv2d_2 (Conv2D)           (None, 10, 10, 4)         292

 conv2d_3 (Conv2D)           (None, 8, 8, 4)           148

 flatten (Flatten)           (None, 256)               0

 dense (Dense)               (None, 10)                2570

=================================================================

- conv2d (Conv2D)             (None, 26, 26, 8)
  de fotos beginnen 28*28 pixels, na de eerste conv2d filter zijn ze 26*26, omdat we padding="valid" hebben gebruikt
  de output is met 2 kleiner geworden omdat er aan alle kanten 1 vanaf is gegaan door de kernel size van 3*3 
  de 8 komt van dat we 8 convolution filters hebben gebruikt

- conv2d_1 (Conv2D)           (None, 24, 24, 8)
  de fotos waren 26*26, omdat we padding="valid" hebben gebruikt
  de output is met 2 kleiner geworden omdat er aan alle kanten 1 vanaf is gegaan door de kernel size van 3*3 
  de 8 komt van dat we 8 convolution filters hebben gebruikt

- max_pooling2d (MaxPooling2D  (None, 12, 12, 8)
  24/2 = 12, 8 omdat de conv2d_1 layer 8 kernels heeft

- dropout (Dropout)           (None, 12, 12, 8)
  output shape blijft hetzelfde

- conv2d_2 (Conv2D)           (None, 10, 10, 4)
  de fotos waren 12*12, omdat we padding="valid" hebben gebruikt
  de output is met 2 kleiner geworden omdat er aan alle kanten 1 vanaf is gegaan door de kernel size van 3*3 
  de 4 komt van dat we 4 convolution filters hebben gebruikt

- conv2d_3 (Conv2D)           (None, 8, 8, 4)
  de fotos waren 10*10, omdat we padding="valid" hebben gebruikt
  de output is met 2 kleiner geworden omdat er aan alle kanten 1 vanaf is gegaan door de kernel size van 3*3 
  de 4 komt van dat we 4 convolution filters hebben gebruikt

- flatten (Flatten)           (None, 256)
  8*8*4 = 256

- dense (Dense)               (None, 10)
  we hebben 10 output mogelijkheden
"""

"""
Opdracht B3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.
- conv2d (Conv2D)             (None, 26, 26, 8)         80
  3*3 =9 (filter size)
  9*8 = 72 (8 filters)
  72+8 = 80 (9 biases)

- conv2d_1 (Conv2D)           (None, 24, 24, 8)         584
  3*3*8 = 72 (de grote van 1 sample)
  72*8 = 576 (8 is aantal filters)
  576+8 = 584 (8 biases)

- max_pooling2d (MaxPooling2D  (None, 12, 12, 8)        0
  geen parameters

- dropout (Dropout)           (None, 12, 12, 8)         0
  geen parameters

- conv2d_2 (Conv2D)           (None, 10, 10, 4)         292
  3*3*8 = 72 (de grote van 1 sample)
  72*4 = 288 (4 filters)
  288+4 = 292 (4 biases)

- conv2d_3 (Conv2D)           (None, 8, 8, 4)           148
  3*3*4 = 36 (de grote van 1 sample)
  36*4 = 144 (4 filters)
  144+4 = 148 (4 biases)

- flatten (Flatten)           (None, 256)               0
  geen parameters

- dense (Dense)               (None, 10)                2570
  256*10 = 2560 (10 output nodes)
  2560+10 = 2570 (10 biases)


"""

"""
# Train the model
"""
batch_size = 4048 # Larger often means faster training, but requires more system memory.
                  # if you get allocation accord
epochs = 200    # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'     # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_B_weights.h5" here.

print ("\n")
print (ViewTools_NN.getColoredText(255,255,0,"Just type CTRL+C anytime if you feel that you've waited for enough episodes."))
print ("\n")
# # NB: validation split van 0.2 ipv 0.1 gaf ook een boost: van 99.49 naar 99.66
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])    
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0])

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat = None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
