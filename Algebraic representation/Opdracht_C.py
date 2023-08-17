"""
MNIST opdracht C: "Only Conv"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je geen dense layer meer.
De output is nu niet meer een vector van 10, maar een
plaatje van 1 pixel groot en 10 lagen diep.

Deze opdracht bestaat uit vier delen: C1 tm C4 (zie verderop)
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import random
from MavTools_NN import ViewTools_NN

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])

# print("show image\n")
# plt.figure()
# plt.imshow(x_test[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train,-2),-2)
y_test = np.expand_dims(np.expand_dims(y_test,-2),-2)

"""
Opdracht C1: 
    
Voeg ALLEEN Convolution en/of MaxPooling2D layers toe aan het onderstaande model.
(dus GEEN dense layers, ook niet voor de output layer)
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0.98.

Voorbeelden van layers:
    layers.Conv2D(getal, kernel_size=(getal, getal))
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

===============================================================================================================
model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #14*14*4
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),    #12*12*4
            keras.layers.MaxPooling2D(pool_size=(3, 3)),                    #4*4*4
            keras.layers.Conv2D(10, kernel_size=(3, 3), padding="same"),    #4*4*10
            keras.layers.MaxPooling2D(pool_size=(4, 4)),                    #1*1*10
        ]
    )
200 epochs: Test accuracy: 0.8518

----------------------------------
eerste convlayer nu 8 filters
 Test accuracy: 0.8960

---------------------------------------------------------------------------------------------------
model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="same"),                
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),    #26*26
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #13*13
            keras.layers.Conv2D(8, kernel_size=(4, 4), padding="valid"),    #10*10
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #5*5
            keras.layers.Conv2D(10, kernel_size=(4, 4), padding="valid"),   #2*2
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #1*1
        ]
    )
    200 epochs Test accuracy: 0.9641
    
--------------------------------------
extra conv2d layer maet padding="same", zodat de output shape hetzelfde blijft
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="same"),                
            keras.layers.Conv2D(4, kernel_size=(3, 3), padding="valid"),    #26*26
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #13*13
            keras.layers.Conv2D(8, kernel_size=(4, 4), padding="valid"),    #10*10
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #5*5
            keras.layers.Conv2D(8, kernel_size=(3, 3), padding="same"),
            keras.layers.Conv2D(10, kernel_size=(4, 4), padding="valid"),   #2*2
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                    #1*1
        ]
    )
    200 epochs: Test accuracy: 0.9682
-------------------------------------------------------------------------------------------
toch de spoiler gebruikt
    de spoiler gebruikt veel meer filters dan ik had,
    ook gebruikte de spoiler een grotere kernel size
    en ik eindigde met een maxpooling layer
    test na 200 epochs
        Test loss: 0.0127
        Test accuracy: 0.9808


    
===============================================================================================================

BELANGRIJK (ivm opdracht D, hierna):  
* Zorg er dit keer voor dat de output van je laatste layer bestaat uit een 1x1 image met 10 lagen.
Met andere woorden: zorg ervoor dat de output shape van de laatste layer gelijk is aan (1,1,10)
De eerste laag moet 1 worden bij het cijfer 0, de tweede bij het cijfer 1, etc.

Tip: Het zou kunnen dat je resultaat bij opdracht B al aardig kunt hergebruiken,
     als je flatten en dense door een conv2D vervangt.
     Om precies op 1x1 output uit te komen kun je puzzelen met de padding, 
     de conv kernal size en de pooling.
     
* backup eventueel na de finale succesvolle training van je model de gegenereerde weights file
  (myWeights.m5). Die kun je dan in opdracht D inladen voor snellere training.
  
  
Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_C.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Conv2D(20, kernel_size=(5, 5)),                #24
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                #12
            keras.layers.Conv2D(20, kernel_size=(3, 3)),                #10
            keras.layers.MaxPooling2D(pool_size=(2, 2)),                #5
            keras.layers.Conv2D(num_classes, kernel_size=(5, 5))        #1
        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht C2: 
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape


 Layer (type)                Output Shape              Param #
=================================================================
- conv2d (Conv2D)             (None, 24, 24, 20)        520
    de fotos beginnen 28*28 pixels, na de eerste conv2d filter zijn ze 24*24, omdat we padding="valid" gebruiken
    de output is met 4 kleiner geworden omdat er aan alle kanten 2 vanaf is gegaan door de kernel size van 5*5 
    de 20 komt van dat we 20 convolution filters hebben gebruikt

- max_pooling2d (MaxPooling2D  (None, 12, 12, 20)       0
    24/2 = 12, 20 omdat de conv2d_1 layer 20 kernels heeft

- conv2d_1 (Conv2D)           (None, 10, 10, 20)        3620
    de fotos waren 12*12 pixels, na dit conv2d filter zijn ze 10*10, omdat we padding="valid" gebruiken
    de output is met 2 kleiner geworden omdat er aan alle kanten 1 vanaf is gegaan door de kernel size van 3*3
    de 20 komt van dat we 20 convolution filters hebben gebruikt

- max_pooling2d_1 (MaxPooling  (None, 5, 5, 20)         0
    10/2 = 5, 20 omdat de conv2d_1 layer 20 kernels heeft

- conv2d_2 (Conv2D)           (None, 1, 1, 10)          5010
    de fotos waren 5*5 pixels, na dit conv2d filter zijn ze 1*1, omdat we padding="valid" gebruiken
    de output is met 4 kleiner geworden omdat er aan alle kanten 2 vanaf is gegaan door de kernel size van 5*5
    de 20 komt van dat we 20 convolution filters hebben gebruikt

=================================================================
"""

"""
Opdracht C3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.
- conv2d (Conv2D)             (None, 24, 24, 20)        520
    5*5 = 25      (groote filter)
    25*20 = 500   (aantal filters)
    500+20 = 520  (aantal biases)

- max_pooling2d (MaxPooling2D  (None, 12, 12, 20)       0
    geen parameters

- conv2d_1 (Conv2D)           (None, 10, 10, 20)        3620
    3*3*20 = 180    (de grote van 1 sample)
    180*20 = 3600   (20 filters)
    3600+20 = 3620  (20 biases)


- max_pooling2d_1 (MaxPooling  (None, 5, 5, 20)         0
    geen parameters

- conv2d_2 (Conv2D)           (None, 1, 1, 10)          5010
    5*5*20 = 500    (de grote van 1 sample)
    500*10 = 5000   (10 filters)
    5000+10 = 5010  (10 biases)
"""

"""
Opdracht C4: 
    
Bij elke conv layer hoort een aantal elementaire operaties (+ en *).
* Geef PER CONV LAYER een berekening van het totaal aantal operaties 
  dat nodig is voor het klassificeren van 1 test-sample.
- conv2d (Conv2D)             (None, 24, 24, 20)        520
    kernel_size=(5, 5)
    voor 1 pixel in output
        25 '*'
        24 '+'               vb 1+2+3+4: 3 plusjes om 4 getallen op te tellen
        1  '+'  voor de bias
    output is 24*24=576
    576*20=11520 voor aantal filters
        25*11520 = 288000 '*'
        25*11520 = 288000 '+'


- conv2d_1 (Conv2D)           (None, 10, 10, 20)        3620
    kernel_size=(3, 3 20)   20 door aantal filters vorige layer
    voor 1 pixel in output
        180 '*'
        179 '+'
        1   '+'  voor de bias
    output is 10*10 = 100
    100*20 = 2000 voor aantal filters
        180*2000 = 360000 '*'
        180*2000 = 360000 '+'

- conv2d_2 (Conv2D)           (None, 1, 1, 10)          5010
    kernel_size=(5, 5, 20)      20 door aantal filters vorige layer
        500 '*'
        499 '+'
        1   '+'  voor de bias
    output inc filters is 1*1*10 = 10
        500*10 = 5000 '*'
        500*10 = 5000 '+'


    

* Op welk aantal operaties kom je uit voor alle conv layers samen?
    layer 1 '*' = 288000
    layer 1 '+' = 288000
    layer 2 '*' = 360000
    layer 2 '+' = 360000
    layer 3 '*' =   5000
    layer 3 '+' =   5000
    ____________________
    totaal =     1306000 elementaire operaties
"""

"""
## Train the model
"""

batch_size = 4096 # Larger means faster training, but requires more system memory.
epochs = 200 # for now

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat straks bij opdracht D ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate) #lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_C_weights.h5" here.
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
print(prediction[0])
print(y_test[0])

# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


print(x_test.shape)

# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)

# baseFilenameForSave=None
# x_test_flat=None
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
