import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

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

#om mee te testen
daysData = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})



#functie om de min en max van de training data te berekenen
def calcTrainingMinMax(data):
  trainingMinMax = []
  for column in range(0,7):
    #[: ,column]
    # : geeft verticaal de hele lijst
    # ,column geeft alleen die column
    #https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    trainingMinMax.append( (min(data[: ,column]), max(data[: ,column])) )
  return trainingMinMax

#functie om data te normaliseren
#gebruikt de min en max van trainingMinMax
def normaliseData(data, minMax):
  for i in range(len(data) ):
    for param in range(0,7):
      #data[param][i] is datapunt aangepast wordt
      #trainingMinMax[param][0]) is de min van param
      #trainingMinMax[param][1] is de max van een param
      data[i][param] = (data[i][param] - minMax[param][0]) / (minMax[param][1] - minMax[param][0])


#functie om k random datapunten in een array te krijgen
#moeten verschillende punten zijn
def pickAllStartingPoints(k, data):
  #lijst met indexes van de random gekozen punten
  startingPointsIndexes = []
  startingPoints = []
  i = 0
  while i < k:
    newIndex = np.random.randint(len(data))
    if not newIndex in startingPointsIndexes:
      startingPointsIndexes.append(newIndex)
      startingPoints.append(data[newIndex])
      i += 1
  return startingPoints

#functie om de kwadraat van de afstand tussen 2 punten met 7 dimenties te berekenen
def calcDistanceSqr2points(DataItem1, DataItem2):
  dist = 0
  #bereken het verschil / de afstand tussen alle colommen
  for i in range(0,7):
    dist = dist + (abs(DataItem1[i] - DataItem2[i])**2)
  return dist

#functie om de afstand van elk punt tot het dichstbijzijnzte centroid te berekenen,
#returnd een array met de dichstbijzijnzte startingpoint en die afstand in een tuple voor elk data punt
def calcDistanceToCentroids(centroids, data):
  #array met de dichstbijzijnzte startingpoint en die afstand in een tuple voor elk data punt
  #array zo lang als data
  nearestKAndDistance = []
  for dataPoint in data:
    #array met tuples met een startingpoint en de kwadraat van de afstand tot dat startingpoint
    dataPointDistances = []
    for centroid in centroids:
      dataPointDistances.append(calcDistanceSqr2points(dataPoint, centroid) )
    #geeft de index van de dichst bijzijnde startingpoint
    nearestK = min(range(0,len(centroids)), key = lambda indexK: dataPointDistances[indexK] )
    nearestKAndDistance.append( (nearestK, dataPointDistances[nearestK]) )
  return nearestKAndDistance

def calcCentroids(k, nearestKAndDistance, data, lastCentroids):
  #som van datapunten gescheiden door dichstbijzijnde centroid
  #[ [0.0]*7]*k geeft een 2d array met k punten met ieder 7 keer 0.0
  sumDataPoints = [ [0.0]*7]*k
  #houd bij hoeveel datapunten bij elk controid horen
  countCentroinds = [0]*k
  for distance, dataPoint in zip(nearestKAndDistance, data):
    sumDataPoints[distance[0] ] += dataPoint
    countCentroinds[distance[0] ] +=1
  
  newCentroids = []
  for sumDataPoint, countcentroid, lastCentroid in zip(sumDataPoints, countCentroinds, lastCentroids):
    if not countcentroid == 0:
      newCentroids.append(sumDataPoint/countcentroid)
    else:
      #lastCentroids wordt meegegeven voor als er gen elkel punt bij een centroid hoort en je dus deeld door 0
      #als dit gebeurt verplaatsen we dit centroid niet
      newCentroids.append(lastCentroid)
  return newCentroids

#functie om te bepalen of de centroids klaar zijn met verplaatsen
def doneMovingCentroids(newCentroids, lastCentroids, movingThreshold):
  for newCentroid, lastCentroid in zip(newCentroids, lastCentroids):
    if movingThreshold < calcDistanceSqr2points(newCentroid, lastCentroid):
      return False
  return True

#functie om de centroids 1 stap te verplaatsen
def iterateCentroids(lastCentroids, data, movingThreshold):
  nearestKAndDistance = calcDistanceToCentroids(lastCentroids, data)
  newCentroids = calcCentroids(len(lastCentroids), nearestKAndDistance, data, lastCentroids)
  if doneMovingCentroids(newCentroids, lastCentroids, movingThreshold):
    return newCentroids
  return iterateCentroids(newCentroids, data, movingThreshold)

def countOccurrence(k, finalNearestKAndDistance):
  counter = [0]*k
  for nearestK in finalNearestKAndDistance:
    counter[ nearestK[0] ] += 1
  return counter

def labelGroups(k, finalNearestKAndDistance, trainingsLabels):
  #array van k*4 grote om voor elke groep bij te houden hoe vaak elk seizoen voorkomt
  #winter, lente, zomer, herfst
  #counter = [[0]*k]*4
  counter = []
  for i in range(0, k):
    counter.append({})

  for NearestK, trainingsLabel in zip(finalNearestKAndDistance, trainingsLabels):
    indexNearestK = NearestK[0]
    myDict = counter[indexNearestK]
    myDict[trainingsLabel] = myDict.get(trainingsLabel, 0) + 1

  #lijst met het vaakst voorkomende label per groep
  groupLabels = []
  for myDict in counter:
    groupLabels.append(max(myDict, key = myDict.get))
  return groupLabels

#berekend de gemiddelde afstand van alle punten tot hun centroid
def aggregateIntraClusterDistance(finalNearestKAndDistance):
  avrDistToCentroids = 0.0
  for NearestK in finalNearestKAndDistance:
    avrDistToCentroids += NearestK[1]
  return avrDistToCentroids/len(finalNearestKAndDistance)

#functie om alle andere functie meerdere keeren te runnen
#k hoeveel verschillende k's er geprobeert worden, van 1 tot k
def runProgramMultipleTimes(k, nTrys, movingThreshold, data):
  minMax = calcTrainingMinMax(trainingData)
  normaliseData(data, minMax)
  
  bestTrys = []
  #k+1 omdat we bij 1 beginnen en niet bij 0
  for currentK in range(1, k+1):
    bestTry = 0.0
    bestFinalNearestKAndDistance = []
    for i in range(0, nTrys):
      startingPoints = pickAllStartingPoints(currentK, data)
      finalCentroids = iterateCentroids(startingPoints, data, movingThreshold)
      finalNearestKAndDistance = calcDistanceToCentroids(finalCentroids, data)
      currentTry = aggregateIntraClusterDistance(finalNearestKAndDistance)
      if bestTry == 0.0 or currentTry < bestTry:
        bestTry = currentTry
        bestFinalNearestKAndDistance = finalNearestKAndDistance
    bestTrys.append(bestTry)
    print("currentK=", currentK, ": ", labelGroups(currentK, bestFinalNearestKAndDistance, trainingsLabels) )
  return bestTrys

def makeGraph(graphData):
  x = []
  for i in range(1, len(graphData)+1):
    x.append(i)

  plt.plot(x, graphData)
  plt.xlabel('k')
  plt.ylabel('intra-cluster distance')
  plt.show()

k = 10
movingThreshold = 0.01
data = trainingData
nTrys = 10

intraClusterDistances = runProgramMultipleTimes(k, nTrys, movingThreshold, data)
makeGraph(intraClusterDistances)

# testK = 1

# minMax = calcTrainingMinMax(data)
# normaliseData(data, minMax)
# startingPoints = pickAllStartingPoints(testK, data)
# finalCentroids = iterateCentroids(startingPoints, data, movingThreshold)
# finalNearestKAndDistance = calcDistanceToCentroids(finalCentroids, data)
# aggregateIntraClusterDistance(finalNearestKAndDistance)
# print( labelGroups(testK, finalNearestKAndDistance, trainingsLabels) )

# TODO
# functie
# eerst juiste labels vergelijken,
# wat is het label wat bij een groep hoort/vaakst voorkomt in een groep [check]

# aggregate intra-cluster distance
# = gemiddelde afstand tot centroid
# -> meerdere keren per k, kies beste
# -> voor meerdere K's > in grafiek