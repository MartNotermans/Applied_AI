import numpy as np
np.random.seed(0)

#random punten is andere opdracht?
def pickStartingPoints(data):
  return ( np.random.randint(len(data)),np.random.randint(len(data)) )

pickStartingPoints(trainingData)