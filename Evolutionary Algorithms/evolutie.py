#Arthur, Lancelot, Gawain, Geraint, Percival, Bors the Younger, Lamorak, Kay Sir Gareth, Bedivere, Gaheris, Galahad, Tristan
#nummers 0 tot 11

import csv
import random
random.seed(0)

import time

from matplotlib import pyplot as plt
import numpy as np
import statistics as stats

class individual:
    def __init__(self, setting, fitness = -1):
        self.setting = setting
        self.fitness = fitness


tableSize = -1

#2d affinity array
affArr = []
names = []

with open('RondeTafel.csv', newline='') as csvfile:
     reader = csv.reader(csvfile, delimiter=';', quotechar='|')
     #om de lines met woorden niet mee te nemen
     next(reader)
     names = next(reader)
     names = names[1:]  #omdat de rij met namen in de csv met een lege waarde begind
     tableSize = len(names)
     for row in reader:
        tempArr = []
        affArr.append(tempArr)
        #row[1:] om de naamweg te halen
        for item in row[1:]:
            tempArr.append(float(item))


def calcScore(per1, per2):
    return affArr[per1][per2] * affArr[per2][per1]

def fitness(volgorde):
    score = 1
    for i in range( len(volgorde[:-1]) ):
        score *= calcScore( volgorde[i], volgorde[i+1] )

    score *= calcScore( volgorde[0], volgorde[-1] )
    return score

def generatePopulation(nPop):
    population = []
    for i in range(nPop):
        startList = [0,1,2,3,4,5,6,7,8,9,10,11]
        random.shuffle(startList)
        population.append(individual(startList))
    return population


def calcPopFitness(population):
    for item in population:
        item.fitness = fitness(item.setting)

def tournament(population, poolSize, nChosen, chanceTopPool):
    chosenOnes = []
    tempPop = population.copy()
    #ga door totdat er genoeg chosenones zijn
    while len(chosenOnes) != nChosen:
        #kiest random indexen tussen 0 en het aantal population in tempPop
        pool = random.sample(range(0,len(tempPop) ), poolSize)
        #sorteerd de pool op hoe goed de fitness was
        pool.sort(reverse=True, key=lambda index: tempPop[index].fitness)
        for i in range(poolSize):
            if random.random() < chanceTopPool: #if gekozen
                index = pool[i]
                chosenOnes.append(tempPop[index])
                tempPop.pop(index)
                break
    return chosenOnes

def testCheckPairOverlap():
    correct = [0,3,6,9] #returnt false
    print(checkPairOverlap(correct))
    incorrect = [0,1,4,8] #returnt true
    print(checkPairOverlap(incorrect))
    incorrect2 = [0,3,7,11] #returnt true
    print(checkPairOverlap(incorrect2))

#de pairs mogen niet overlappen, omdat elk getal maar 1x mag voorkomen
def checkPairOverlap(chosenPairs):
    for i in range(len(chosenPairs)-1 ):
        if chosenPairs[i+1] - chosenPairs[i] == 1:
            return True
    if chosenPairs[0] == 0 and chosenPairs[-1] == tableSize-1:
        return True
    return False

#position-based crossover 
def crossover(parent1, parent2):
    #gekozen getal is de linker helft van een pair
    #Select a set of position from one parent at random.
    chosenPairs = random.sample(range(0,tableSize), 2)
    chosenPairs.sort()
    while checkPairOverlap(chosenPairs):
        chosenPairs = random.sample(range(0,tableSize), 2)
        chosenPairs.sort()

    child1 = crossoverWork(parent1, parent2, chosenPairs)
    child2 = crossoverWork(parent2, parent1, chosenPairs)
    return (child1, child2)

def testCrossoverWork():
    parent1 = [0,1,2,3,4,5,6,7,8,9,10,11]
    parent2 = [11,10,9,8,7,6,5,4,3,2,1,0]
    chosenPairs = [1,5]
    child1 = crossoverWork(parent1, parent2, chosenPairs)
    child2 = crossoverWork(parent2, parent1, chosenPairs)

    if child1 == [11,1,2,10,9,5,6,8,7,4,3,0] and child2 == [0,10,9,1,2,6,5,3,4,7,8,11]:
        return True
    return False

def crossoverWork(parent1, parent2, chosenPairs):
    child = [-1]*tableSize

    verplaatsteNummers = []
    #de pairs van de parent copieeren naar de child
    #Produce a proto-child by copying the cities on these positions into the corresponding position of the proto-child.
    for pair in chosenPairs:
        child[pair] = parent1[pair]
        verplaatsteNummers.append(parent1[pair])
        #%tableSize omdat de tafel rond is en je dus doortelt
        child[(pair+1)%tableSize] = parent1[(pair+1)%tableSize]
        verplaatsteNummers.append(parent1[(pair+1)%tableSize])

    #Delete the cities which are already selected from the second parent.The resulting sequence of cities contains the cities the proto-child needs.
    # Removing elements present in other list
    # Using filter() + lambda
    parent2MinVerplaatsteNummers = list(filter(lambda i: i not in verplaatsteNummers, parent2) )

    #Place the cities into the unfixed position of the proto-child from left to right according to the order of the sequence to produce one offspring
    indexParent2MinVerplaatsteNummers = 0
    for i in range(0,len(child) ):
        if child[i] == -1:
            child[i] = parent2MinVerplaatsteNummers[indexParent2MinVerplaatsteNummers]
            indexParent2MinVerplaatsteNummers+=1
    
    return child

#swap mutatie
def mutation(parent):
    sample = random.sample(range(0,tableSize), 2)
    parent[sample[0]], parent[sample[1]] = parent[sample[1]], parent[sample[0]]


def genetischAlgoritme(populationSize, mutationChance, nParents, poolSizePar, chanceTopPoolPar, poolSizeSur, chanceTopPoolSur, population, plotPointsFitness, plotPointsEpochs, runTime):
    print("start")
    bestSetting = None
    epoch = 0
    calcPopFitness(population)
    best = max(population, key=lambda pop: pop.fitness)
    print(best.setting, best.fitness, "epoch: ", epoch)
    plotPointsFitness.append(best.fitness)
    plotPointsEpochs.append(epoch)

    start_time = time.time()
    while time.time() - start_time < (runTime):    #3 minuten
        epoch+=1
        calcPopFitness(population)
        #kies parents
        chosenOnes = tournament(population, poolSizePar, nParents, chanceTopPoolPar)
        random.shuffle(chosenOnes)
        offspring = []

        #doe de crossover
        for i in range(0,len(chosenOnes)-1):
            children = crossover(chosenOnes[i].setting, chosenOnes[i+1].setting)
            offspring.append(individual(children[0]) )
            offspring.append(individual(children[1]) )

        #doe mischien de mutation
        for i in range(0,len(offspring)):
            if random.random() < mutationChance:
                mutation(offspring[i].setting)

        calcPopFitness(offspring)

        population.extend(offspring)
        #kies survivals
        #todo: bovenste 5% niet meenemen in tournement, omdat er nu een kans is dat je de beste per ongeluk weggooit
        population = tournament(population, poolSizeSur, populationSize, chanceTopPoolSur)

        best = max(population, key=lambda pop: pop.fitness)
        bestSetting = best.setting
        print(best.setting, best.fitness, "epoch: ", epoch)
        plotPointsFitness.append(best.fitness)
        plotPointsEpochs.append(epoch)
    
    return bestSetting


def maakPlot(plotPointsEpochs, plotPointsFitness):
    plt.plot(plotPointsEpochs, plotPointsFitness)
    plt.xlabel("epochs")
    plt.ylabel("fitness")
    plt.show()

def generateWinString(winList):
    #arthur vooraan lijst
    while winList[0] != 0:
        winList.append(winList.pop(0) )

    winstring = ""
    for i in range(0,len(winList)):
        winstring += names[winList[i] ]#testen
        winstring += " ("
        winstring += str(affArr[winList[i] ][winList[(i+1)%tableSize] ] )
        winstring += "x"
        winstring += str(affArr[winList[(i+1)%tableSize] ][winList[i] ] )
        winstring += ") "
    return winstring

populationSize = 10000
mutationChance = 0.2
nParents = 1000

#voor tournement parents
poolSizePar = 100
chanceTopPoolPar = 0.6

#voor tournement survivals
poolSizeSur = 50
chanceTopPoolSur = 0.8


myPopulation = generatePopulation(populationSize)

plotPointsFitness = []
plotPointsEpochs = []

#hoelang het algoritme runt in seconden
runTime = 60 #180 = 3minuten

bestSetting = genetischAlgoritme(populationSize, mutationChance, nParents, poolSizePar, chanceTopPoolPar, poolSizeSur, chanceTopPoolSur, myPopulation, plotPointsFitness, plotPointsEpochs, runTime)
maakPlot(plotPointsEpochs, plotPointsFitness)
print(generateWinString(bestSetting) )