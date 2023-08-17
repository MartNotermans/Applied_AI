"""
MNIST opdracht D: "Localisatie"      (by Marius Versteegen, 2021)
Bij deze opdracht passen we het in opdracht C gevonden model toe om
de posities van handgeschreven cijfers in een plaatje te localiseren.
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

print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Now, for a test on the end, let's create a few large images and see if we can detect the digits
# (we do that here because the dimensions of the images are not extended yet)
largeTestImage = np.zeros(shape=(1024,1024),dtype=np.float32)
# let's fill it with a random selection of test-samples:
nofDigits = 40
for i in range(nofDigits):
    # insert an image
    digitWidth  = x_test[0].shape[0]
    digitHeight = x_test[0].shape[1]
    xoffset = random.randint(0, largeTestImage.shape[0]-digitWidth)
    yoffset = random.randint(0, largeTestImage.shape[1]-digitWidth)
    largeTestImage[xoffset:xoffset+digitWidth,yoffset:yoffset+digitHeight] = x_test[i][0:,0:]
    
largeTestImages = np.array([largeTestImage])  # for now, I only test with one large test image.

print("show largeTestImage\n")
plt.figure()
plt.imshow(largeTestImages[0])
plt.colorbar()
plt.grid(False)
plt.show()
matplotlib.image.imsave('largeTestImage.png',largeTestImages[0])


# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
largeTestImages = np.expand_dims(largeTestImages, -1)  # one channel per color component in lowest level. In this case, thats one.

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
Opdracht D1: 

Kopieer in het onderstaande model het model dat resulteerde
in opdracht C. Gebruik evt de weights-file uit opdracht C, om trainingstijd in te korten.

Bestudeer de output files: largeTestImage.png en de Featuremaps pngs.
* Leg uit hoe elk van die plaatjes tot stand komt en verklaar aan de hand van een paar
  voorbeelden wat er te zien is in die plaatjes.

als je 1 nummer per input hebt, zal 1 output oplighten,
omdat er nu meerdere nummers in de input zitten geeft elk plaatje een highlight voor een ander nummer,
A_1 highlights 0, A_2 highlights 1 enz.
op de featuremap...A_1 zie je dat de nullen gehighlight worden, bij alle nummers is iets te zien,
maar de nullen zijn het felst

* Als het goed is zie je dat de pooling layers in grote lijnen bepalen hoe groot de 
  uiteindelijke output bitmaps zijn. Hoe werkt dat?

de pooling layers zijn speciafik gekozen om van een 28*28 input een 1*1*10 output te maken
als je een grotere input gebruikt wort die ook kleiner, maar minder klein
van 1024*1024 naar 250*250*10
de pooling layers bepalen het meest omdat die de output shape door 2 delen,
en de Conv2D layers de output met 2 of 4 kleiner maken
de pooling layers hebben een grotere inpact op de output shape
"""

"""
Opdracht D2:
    
* Met welke welke stappen zou je uit de featuremap plaatjes lijsten kunnen 
genereren met de posities van de cijfers?

je kan beginnen met in A_1 kijken waar een cijfer is en opslaan welke pixels bij 1 cijfer horen (als daaromheen alleen achtergrond zit)
voor een cijfer wat je wil weten kun je in de 10 output images kijken welke op die plek de grootste highlight heeft,
de output image met de grootste highlight laat zien welk cijfer het was
"""

"""
Opdracht D3: 

* Hoeveel elementaire operaties zijn er in totaal nodig voor de convoluties voor 
  onze largeTestImage? (Tip: kijk terug hoe je dat bij opdracht C berekende voor 
  de kleinere input image)

=================================================================
 conv2d_3 (Conv2D)           (None, 1020, 1020, 20)    520
    kernel_size=(5, 5)
    voor 1 pixel in output
        25 '*'
        24 '+'               vb 1+2+3+4: 3 plusjes om 4 getallen op te tellen
        1  '+'  voor de bias
    output is 1020*1020 = 1040400
    1040400*20 = 20808000 voor aantal filters
        25*20808000 = 520200000 '*'
        25*20808000 = 520200000 '+'

 conv2d_4 (Conv2D)           (None, 508, 508, 20)      3620
    kernel_size=(3, 3 20)   20 door aantal filters vorige layer
    voor 1 pixel in output
        180 '*'
        179 '+'
        1   '+'  voor de bias
    output is 508*508 = 258064
    258064*20 = 5161280 voor aantal filters
        180*5161280 = 929030400 '*'
        180*5161280 = 929030400 '+'

 conv2d_5 (Conv2D)           (None, 250, 250, 10)      5010
    kernel_size=(5, 5, 20)      20 door aantal filters vorige layer
        500 '*'
        499 '+'
        1   '+'  voor de bias
    output inc filters is 250*250*10 = 625000
        500*625000 = 312500000 '*'
        500*625000 = 312500000 '+'
 
        
    layer 1 '*' = 520200000
    layer 1 '+' = 520200000
    layer 2 '*' = 929030400
    layer 2 '+' = 929030400
    layer 3 '*' = 312500000
    layer 3 '+' = 312500000
    ____________________
    totaal =     3523460800 elementaire operaties

=================================================================

* Hoe snel zou een grafische kaart met 1 Teraflops dat theoretisch kunnen uitrekenen?
1 Teraflops is one trillion (10^12) floating-point operations per second
in 1 milisec 10^9 floating-point operations
3523460800/10^9 = 3.523 miliseconden

* Waarom zal het in de praktijk wat langer duren?
dingen in memory verschuiven, overhead
en we hebben de max pooling layers genegeerd, die kosten ook tijd
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            #spoiler C
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
## Train the model
"""

batch_size = 4096 # Larger means faster training, but requires more system memory.
epochs = 250 # for now

bInitialiseWeightsFromFile = True # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat ook de afwezigheid van een cijfer een valide mogelijkheid is.
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

baseFilenameForSave=None
x_test_flat=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

lstWeights=[]
for nLanger in range(len(model.layers)):
    lstWeights.append(model.layers[nLayer].get_weights())

# rebuild model, but with larger input:
model2 = buildMyModel(largeTestImages[0].shape)
model2.summary()

for nLayer in range(len(model2.layers)):
      model2.layers[nLayer].set_weights(model.layers[nLayer].get_weights())

prediction2 = model2.predict(largeTestImages)
baseFilenameForSave='FeaturemapOfLargeTestImage_A_'
ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model2, None, largeTestImages,  0, baseFilenameForSave)
