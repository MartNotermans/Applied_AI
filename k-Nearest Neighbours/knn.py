import numpy as np
#import math

#import the training data
#test = np.fromfile("test.csv")
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

#import the validation data
validationData = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validationDates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
validationLabels = []

for date in validationDates:
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

#varible om de min en max van de training data bij te houden
trainingMinMax = []
#functie om de min en max van de training data te berekenen
def calcTrainingMinMax(data):
  for column in range(0,7):
    #[: ,column]
    # : geeft verticaal de hele lijst
    # ,column geeft alleen die column
    #https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    trainingMinMax.append( (min(data[: ,column]), max(data[: ,column])) )

#functie om data te normaliseren
#gebruikt de min en max van trainingMinMax
def normaliseData(data):
  for i in range(len(data) ):
    for param in range(0,7):
      #data[param][i] is datapunt aangepast wordt
      #trainingMinMax[param][0]) is de min van param
      #trainingMinMax[param][1] is de max van een param
      data[i][param] = (data[i][param] - trainingMinMax[param][0]) / (trainingMinMax[param][1] - trainingMinMax[param][0])

#functie om de afstand tussen 2 punten met 7 dimenties te berekenen
def calcDistance2points(trainingDataItem, validationDataItem):
  dist = 0
  #bereken het verschil / de afstand tussen alle colommen
  for i in range(0,7):
    dist = dist + (abs(trainingDataItem[i] - validationDataItem[i])**2)
  #dist = math.sqrt(dist)
  return dist

#fusctie om de afstand van elk punt uit een set te berekenen met elk punt uit de training data
def calcDistanceSet(validationData):
  #array met voor elk validation item een array met de distance tot de training data  
  validationDistances = []
  for validationItem in validationData:
    #array van 1 validation item met de afstanden tot alle training data met het bijbehorende label in een tuple
    validationItemDistances = []
    #loop door 2 arrays tegelijkertijd
    for trainingItem, label in zip(trainingData, trainingsLabels):
      validationItemDistances.append( (calcDistance2points(trainingItem, validationItem), label) )
    #validationItemDistances.sort()#sorteer van klein naar groot
    validationDistances.append(validationItemDistances)
  #print(validationDistances)
  return validationDistances

#functie om de k kleinste afstanden te zoeken en terug te geven als een array
def findKnn(k, validationItemDistances):
  Knn = validationItemDistances[0:k]
  Knn.sort()
  #distance is een tuple met de distance en het label
  for distance in validationItemDistances[k:]:
    #vergelijk de nieuwe afstand met het grootste item in Knn
    #knn[-1][0], [-1] is laatsete/grootste item, [0] is de afstanduit de tuple
    if distance[0] < Knn[-1][0]:
      for i in range(k):
        if distance[0] < Knn[i][0]:
          Knn.insert(i, distance)
          Knn.pop()
          break

  return Knn

#functie om het label wat het vaakst voorkomt in een array terug te geven
#wat als er meerdere zeizoenen de meeste zijn
def findMostFrequentLabel(Knn):
  dict = {}
  for nn in Knn:
    dict[nn[1]] = dict.get(nn[1], 0) + 1
  return max(dict, key = dict.get)

#functie om te controleren hoeveel labels corect berekend zijn
# #vergelijk de gekozen labels met de correcte labels, returnt het aantal corecte labels
def calcValidationKnn(k, validationDistances):
  nCorrectLabels = 0
  for distanceArray, correctlabel in zip(validationDistances, validationLabels):
    if findMostFrequentLabel( findKnn(k, distanceArray) ) == correctlabel:
      #print(findMostFrequentLabel( findKnn(k, distanceArray) ))
      nCorrectLabels = nCorrectLabels+1

  return nCorrectLabels


calcTrainingMinMax(trainingData)
normaliseData(trainingData)

#k = 62 geeft 63% correct
normaliseData(validationData)

#for k in range(1, 100):
#  print(k, " ", calcValidationKnn(k, calcDistanceSet(validationData) ))

k = 62
print(k, " ", calcValidationKnn(k, calcDistanceSet(validationData) ))