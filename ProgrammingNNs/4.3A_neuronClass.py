class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias= bias
        self.activation = None
        self.neurons = []

    def setActivation(self, activation_):
        activation = activation_

    def setNeurons(self, neurons_):
        self.neurons = neurons_

    def compute(self, inputs):
        total = 0
        for i in range(len(self.neurons)):
            total += self.neurons[i].compute(inputs)*self.weights[i]
        total += self.bias
        return total


class inputNeuron(neuron):
    def __init__(self, inputIndex):
        self.inputIndex = inputIndex

    def compute(self, inputs):
        return inputs[self.inputIndex]

#om in een keer een xor object te kunnen gebruiken
class groupNeuron(neuron):
    #de input van de group
    groupInput = []

    def __init__(self, neuronNetwork):
        #de onderdelen van het neuronNetwork
        self.neuronNetwork = neuronNetwork

    def setNeurons(self, neurons_):
        self.groupInput = neurons_

    def compute(self, inputs):
        neuronOutputList = []
        for i in range(len(self.groupInput)):
            neuronOutputList.append(self.groupInput[i].compute(inputs))
        
        return self.neuronNetwork.compute(neuronOutputList)



#opdracht 4.1, 3 input nor gate
inputs = [
    inputNeuron(0),
    inputNeuron(1),
    inputNeuron(2)
]

threeInputNORgate = neuron([-1, -1, -1], 1)
threeInputNORgate.setActivation(lambda x : x>=1)
threeInputNORgate.setNeurons(inputs)

print(threeInputNORgate.compute([0,0,0]))



#opdracht 4.2, neural adder

inputA = inputNeuron(0),
inputB = inputNeuron(1),
inputC = inputNeuron(2)

#xor1
ORgate_xor = neuron([2,2],-1)
ORgate_xor.setActivation(lambda x : x>=1)
ORgate_xor.setNeurons([inputA, inputB])
NANDgate_xor = neuron([1,1],2)
NANDgate_xor.setActivation(lambda x : x>=1)
NANDgate_xor.setNeurons([inputA, inputB])
ANDgate_xor = neuron([1,1],-1)
ANDgate_xor.setActivation(lambda x : x>=1)
ANDgate_xor.setNeurons([ORgate_xor, NANDgate_xor])

XOR1 = groupNeuron(ANDgate_xor)
XOR1.setNeurons([inputA, inputB])
XOR2 = groupNeuron(ANDgate_xor)
XOR2.setNeurons([XOR1, inputC])

ANDgate1 = neuron([1,1],-1)
ANDgate1.setActivation(lambda x : x>=1)
ANDgate1.setNeurons([XOR1, inputC])

ANDgate2 = neuron([1,1],-1)
ANDgate2.setActivation(lambda x : x>=1)
ANDgate2.setNeurons([inputA, inputB])

ORgate = neuron([2,2],-1)
ORgate.setActivation(lambda x : x>=1)
ORgate.setNeurons([ANDgate1, ANDgate2])


print("xor1: ", XOR1.compute([0,0]))


inputs = [0,0,0]
outputS = XOR2.compute(inputs)
outputC = ORgate.compute(inputs)
print(outputS, outputC)


