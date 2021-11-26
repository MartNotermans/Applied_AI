import numpy as np

#colum number is 1 bigger than array place
trainingData = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

trainingDates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
trainingsLabels = []

for date in trainingDates:
  if date < 20000301:
    trainingsLabels.append('winter')
  elif 20000301 <= date < 20000601:
    trainingsLabels.append('lente')
  elif 20000601 <= date < 20000901:
    trainingsLabels.append('zomer')
  elif 20000901 <= date < 20001201:
    trainingsLabels.append('herfst')
  else: # from 01-12 to end of year
    trainingsLabels.append('winter')

#validation data
validationData = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validationDates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
validationLabels = []

for date in validationLabels:
  if date < 20010301:
    validationLabels.append('winter')
  elif 20010301 <= date < 20010601:
    validationLabels.append('lente')
  elif 20010601 <= date < 20010901:
    validationLabels.append('zomer')
  elif 20010901 <= date < 20011201:
    validationLabels.append('herfst')
  else: # from 01-12 to end of year
    validationLabels.append('winter')



#varible + functie om de min en max van de training data bij te houden
trainingMinMax = []
def calcTrainingMinMax(data):
  for colum in range(0,7):
    trainingMinMax.append( (min(data[colum]), max(data[colum])) )

#functie om alle data te normaliseren
def normaliseData(data):
  for i in range(len(data) ):
    for param in range(0,7):
      #data[param][i] is datapunt aangepast wordt
      #trainingMinMax[param][0]) is de min van param
      #trainingMinMax[param][1] is de max van een param
      data[i][param] = (data[i][param] - trainingMinMax[param][0]) / (trainingMinMax[param][0] - trainingMinMax[param][1])

def calcDistance2points(trainingDataItem, validationDataItem):
  dist = []
  #bereken het verschil / de afstand tussen alle colommen
  for i in range(0,7):
    dist.append(abs(trainingDataItem[i] - validationDataItem[i]) )
  return dist

#array met voor elk validation item een array met de distance tot de training data     
validationDistances = []
def calcDistanceSet():
  for validationItem in validationData:
    #array van 1 validation item met de afstanden tot de training data
    validationItemDistances = []
    for trainingItem in trainingData:
      validationItemDistances.append( calcDistance2points(trainingItem, validationItem) )
    validationItemDistances.sort()#sorteer van klein naar groot
    validationDistances.append(validationItemDistances)

calcTrainingMinMax(trainingData)
normaliseData(trainingData)
normaliseData(validationData)
calcDistanceSet()

