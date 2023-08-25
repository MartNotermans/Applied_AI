import numpy as np
import random
random.seed(0)

from statistics import mean

#binary step function
def binaryStep(input):
    if input >= 1:
        return 1
    return 0

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def derivativeSigmoide(input):
    return sigmoid(input) * (1-sigmoid(input) )

activation = sigmoid
activationDerivetive = derivativeSigmoide


idNumber = 0
def newID():
    global idNumber
    ID = idNumber
    idNumber += 1
    return ID

class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias= bias
        self.neurons = []
        self.ID = newID()


    #kies de gebruikte activation code, krijgt ook de derivative van de activation voor de update functie
    def setActivation(self, activation_, activationDerivetive_):
        self.activation = activation_
        self.activationDerivetive = activationDerivetive_

    #de neurons voor deze neuron
    def setNeurons(self, neurons_):
        self.neurons = neurons_
        
    def weightedSum(self, inputs):
        total = 0
        for i in range(len(self.neurons)):
            total += self.neurons[i].compute(inputs)*self.weights[i]
        total += self.bias
        return total

    #werkt nu alleen voor de output neurons
    #formule 4.14 uit de reader
    def getOutputDelta(self, inputs, desiredOutput):
        output = self.compute(inputs)
        #∆k = g'(ink)(yk −ak) 
        return self.activationDerivetive(self.weightedSum(inputs))*(desiredOutput-output)
    
    #formule 4.17 uit de reader
    #∆j =g'(inj) * ∑Wj,p*∆p
    #              p
    def getHiddenDelta(self, inputs, weightsNextNeurons, deltasNextNeurons):
        #g'(inj)
        ginj = self.activationDerivetive(self.weightedSum(inputs))


        #∑Wj,p*∆p
        #p
        sum = 0
        for p in range(len(deltasNextNeurons)):
            sum += weightsNextNeurons[p] * deltasNextNeurons[p]

        #∆j
        return ginj * sum

    #update de weights dmv de delta rule
    #krijgt de deltas mee
    def update(self, inputs, learningRate, delta):
        for j in range(len(self.weights)):
            #Wj is de oude weight
            Wj = self.weights[j]
            #aj is de input van deze neuron bij Wj
            aj = self.neurons[j].compute(inputs)

            #formule 4.15 uit de reader
            #wj,k = w'j,k +η*aj*∆k
            #delta = g'(ink)(yk −ak)
            self.weights[j] = Wj+(learningRate*(delta*aj) )

        self.bias = self.bias+(learningRate*delta)

    #update een single neuron
    #bereken de delta met de desired output
    def updateSingleNeuronNetwork(self, inputs, desiredOutput, learningRate):
        delta = self.getOutputDelta(inputs, desiredOutput)
        self.update(inputs, learningRate, delta)



    def compute(self, inputs):
        return self.activation(self.weightedSum(inputs))


class inputNeuron(neuron):
    def __init__(self, inputIndex):
        self.inputIndex = inputIndex

    def compute(self, inputs):
        return inputs[self.inputIndex]

#om in een keer een xor object te kunnen gebruiken
class groupNeuron(neuron):
    #de input van de group
    groupNeurons = []

    def __init__(self, neuronNetwork):
        #de onderdelen van het neuronNetwork
        self.neuronNetwork = neuronNetwork

    #de neurons voor deze neuron
    def setNeurons(self, neurons_):
        self.groupNeurons = neurons_

    def compute(self, inputs):
        neuronOutputList = []
        for i in range(len(self.groupNeurons)):
            neuronOutputList.append(self.groupNeurons[i].compute(inputs))
        
        return self.neuronNetwork.compute(neuronOutputList)

#gebruikt in class neuralNetwork om de neurons te initialise met random weights and biases tussen -1 en 1
def randomWeights(aantalWeights):
    weights = []
    for i in range(aantalWeights):
        weights.append(random.uniform(-1, 1) )
    return weights

#used to make a fully connected network
class neuralNetwork():
    #networkShape is een lijst met hoeveel neuronen er in een bepaalde layer zitten
    def __init__(self, networkShape, activation, activationDerivetive) -> None:
        self.network = []
        
        inputNeurons = []
        #het aantal input neurons
        for i in range(networkShape[0]):
            inputNeurons.append(inputNeuron(i))
        self.network.append(inputNeurons)

        #go door de normale layers heen, skip de input layer
        for nlayer in range(1, len(networkShape)):
            currentLayer = []
            #het aantal neuronen in een bapaalde layer
            for i in range(networkShape[nlayer]):
                #randomWeights(networkShape[nlayer-1]) is het aantal weights is het aantal neurons in de vorige layer
                myNeuron = neuron(randomWeights(networkShape[nlayer-1]),random.uniform(-1, 1) )
                myNeuron.setActivation(activation, activationDerivetive)
                myNeuron.setNeurons(self.network[nlayer-1])
                currentLayer.append(myNeuron)
            self.network.append(currentLayer)

    #get de weights die van node weg gaan
    #neuronIndex is de index van de node waarvan je wilt weten welke weights daarvan weg gaan
    def getWeightsNextNeurons(self, neuronIndex, LayerIndex):
        weights = []
        nextLayerIndex = LayerIndex+1
        #loop door de next layer heen
        for i in range(len(self.network[nextLayerIndex]) ):
            #get de weigt van de neuron in de volgende layer naar de neuron die je wou weten
            weights.append( self.network[nextLayerIndex][i].weights[neuronIndex] )
            
        return weights

    def Backpropagation(self, inputs, desiredOutputs, learningRate):       
        deltasNextLayer = []
        outputdeltas = []
        for outputNeuron in range(len(self.network[-1]) ):
            desiredOutput = desiredOutputs[outputNeuron]
            #output = self.network[-1][outputNeuron].compute(inputs)

            outputdeltas.append( self.network[-1][outputNeuron].getOutputDelta(inputs, desiredOutput) )
        deltasNextLayer = outputdeltas
            
        #for loop gaat van de een na laatste layer tot aan de 2de layer, skipt de inputen output layers
        for layerIndex in reversed(range(1, len(self.network)-1 ) ):
            deltasCurrentLayer = []
            #for loop door de nodes in en bepaalde layer
            for neuronIndex in range(len( self.network[layerIndex]) ):
                #bereken de delta en voeg die toe aan deltasCurrentLayer
                weightsNextNeurons = self.getWeightsNextNeurons(neuronIndex, layerIndex)
                currentNeuron = self.network[layerIndex][neuronIndex]
                delta = currentNeuron.getHiddenDelta(inputs, weightsNextNeurons, deltasNextLayer)
                deltasCurrentLayer.append(delta)

                #de weights updaten!! :)
                currentNeuron.update(inputs, learningRate, delta)


            deltasNextLayer = deltasCurrentLayer

    def compute(self, inputs):
        outputs = []
        for i in range(len(self.network[-1]) ):
            outputs.append(self.network[-1][i].compute(inputs) )
        return outputs

        


#opdracht 4.1, 3 input nor gate
inputs = [
    inputNeuron(0),
    inputNeuron(1),
    inputNeuron(2)
]

threeInputNORgate = neuron([-1, -1, -1], 1)
#None omdat we de derivative van de activation functie hier niet gebruiken
threeInputNORgate.setActivation(lambda x : x>=1, None)
threeInputNORgate.setNeurons(inputs)

print("three Input NOR gate: ", int(threeInputNORgate.compute([0,0,0])) )



#opdracht 4.2, neural adder
inputA = inputNeuron(0)
inputB = inputNeuron(1)
inputC = inputNeuron(2)

#xor
ORgate_xor = neuron([2,2],-1)
ORgate_xor.setActivation(activation, activationDerivetive)
ORgate_xor.setNeurons([inputA, inputB])
NANDgate_xor = neuron([-1,-1],2)
NANDgate_xor.setActivation(activation, activationDerivetive)
NANDgate_xor.setNeurons([inputA, inputB])
ANDgate_xor = neuron([1,1],-1)
ANDgate_xor.setActivation(activation, activationDerivetive)
ANDgate_xor.setNeurons([ORgate_xor, NANDgate_xor])

XOR1 = groupNeuron(ANDgate_xor)
XOR1.setNeurons([inputA, inputB])
XOR2 = groupNeuron(ANDgate_xor)
XOR2.setNeurons([XOR1, inputC])

ANDgate1 = neuron([1,1],-1)
ANDgate1.setActivation(activation, activationDerivetive)
ANDgate1.setNeurons([XOR1, inputC])

ANDgate2 = neuron([1,1],-1)
ANDgate2.setActivation(activation, activationDerivetive)
ANDgate2.setNeurons([inputA, inputB])

ORgate = neuron([2,2],-1)
ORgate.setActivation(activation, activationDerivetive)
ORgate.setNeurons([ANDgate1, ANDgate2])

#xor test
#xor truth table
# A B Q
# 0 0 0
# 0 1 1
# 1 0 1
# 1 1 0
xorTestInputA = inputNeuron(0)
xorTestInputB = inputNeuron(1)
XOR3 = groupNeuron(ANDgate_xor)
XOR3.setNeurons([xorTestInputA, xorTestInputB])
#print("xor test: ", XOR3.compute([1,1]))

#gemaakt om de adder voledig te kunnen testen
#werkt alleen met binary step activation
def testAdder():
    inputs = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
    ]

    outputs = [
        [0,0],
        [0,1],
        [0,1],
        [1,0],
        [0,1],
        [1,0],
        [1,0],
        [1,1]
    ]
    for i in range(len(inputs) ):
        if not(ORgate.compute(inputs[i]) == outputs[i][0] and XOR2.compute(inputs[i]) == outputs[i][1]):
            print("here")
            return False
    return True

print("full test adder result: ", testAdder())

#bitwise adder
#gemaakt als test
#and &
#or |
#xor ^
#werkt voor de hele truth table van de neural adder
operatorInputA = 1
operatorInputB = 1
operatorInputC = 1

operatorOutpitS = None
operatorOutpitC = None

operatorOutpitS = ((operatorInputA^operatorInputB)^operatorInputC)
operatorOutpitC = (((operatorInputA^operatorInputB)&operatorInputC)|(operatorInputA&operatorInputB))
#print("bitwise adder: ", operatorOutpitC, operatorOutpitS)





#3 input nor gate met getrainde weights
inputNeurons = [
    inputNeuron(0),
    inputNeuron(1),
    inputNeuron(2)
]

weights = [random.uniform(-1,1),
          random.uniform(-1,1),
          random.uniform(-1,1)]
bias = random.uniform(-1,1)

threeInputNORgate = neuron(weights, bias)
#None omdat we de derivative van de activation functie hier niet gebruiken
threeInputNORgate.setActivation(activation, activationDerivetive)
threeInputNORgate.setNeurons(inputNeurons)

inputs = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
    ]

desiredOutputThreeInputNORgate = [1,0,0,0,0,0,0,0]

for i in range(10000):
    index = random.randrange(0, len(inputs), 1)
    threeInputNORgate.updateSingleNeuronNetwork(inputs[index], desiredOutputThreeInputNORgate[index], 0.2)

for i in range(0, len(inputs)):
    print(threeInputNORgate.compute(inputs[i]) )

print("\n===================================================\n")

#4.3 D
def flowerNameToNumber(name):
    if name == "Iris-setosa":
        return [1,0,0]
    if name == "Iris-versicolor":
        return [0,1,0]
    if name == "Iris-virginica":
        return [0,0,1]

file = open("iris.data", "r")

#data splitsen in training en testdata
trainingData = []
testData = []
for line in file:
    line = line.split(",")
    for i in range(4):
        line[i] = float(line[i])
    #haal de enter weg aan het einde van de line
    line[-1] = line[-1][:-1]

    #normalise data using min and max values
    line[0] = (line[0] - 4.3)/(7.9-4.3)
    line[1] = (line[1] - 2.0)/(4.4-2.0)
    line[2] = (line[2] - 1.0)/(6.9-1.0)
    line[3] = (line[3] - 0.1)/(2.5-0.1)

    #tuple met de inputs and the desired output
    tupleData = (line[:4], flowerNameToNumber(line[-1]) )
    #75% gaat naar trainingData, 25% gaat naar testData
    if random.random() < 0.75:
        trainingData.append(tupleData)
    else:
        testData.append(tupleData)


irisNetwork = neuralNetwork([4,4,4,3], activation, activationDerivetive)
#trainingDataInputs = random.choice(trainingData)[:4]

#formule 4.7 uitde reader
#C(~w) = MSE = (1/2n)*∑|yi-a(xi)|^2
def calcCost(outputs, desiredOutputs):
    sum = 0
    for out, desiredout in zip(outputs, desiredOutputs):
        sum += (desiredout - out)**2

    return 1/(2*len(outputs) ) * sum

def calcAllCosts(testData, printing):
    costs = []
    for input, desiredOutput in testData:
        outputs = irisNetwork.compute(input)
        cost = calcCost(outputs, desiredOutput)
        costs.append( cost )

        if printing:
            print(cost, outputs, desiredOutput)

    return costs

def percentCorrect(testData):
    nCorrect = 0
    for input, desiredOutputs in testData:
        outputs = irisNetwork.compute(input)
        if outputs.index(max(outputs)) == desiredOutputs.index(max(desiredOutputs)):
            nCorrect += 1
    return (nCorrect/len(testData)*100)

#train network
trainTime = 100000
for i in range(trainTime):
    #random.choice(trainingData)[:4] kiest een random line uit de trainingsdata
    # en geeft de eerste 4 elementen, dit zijn de inputs
    inputs, desiredOutputs = random.choice(trainingData)
    irisNetwork.Backpropagation(inputs, desiredOutputs, 0.3)

    if i % (trainTime/100) == 0:
        print("prog", i/trainTime*100, "%")
        print("mean cost", mean(calcAllCosts(testData, False) ) )
        print("% correct", percentCorrect(testData) )

calcAllCosts(testData, True)