# For each context length N (up to 38), linearly interpolates the probabilities conditioned contexts of lengths up to N to minimize surprisal.


import torch

import sys

infix = "-tokens-whole"

from config import NATURALSTORIES_PATH

with open(NATURALSTORIES_PATH+"/merged"+infix+".txt", "r") as inFile:
  data = [x.split("\t") for x in inFile.read().strip().split("\n")]
columnNames = data[0]
header = dict(zip(data[0], range(len(data[0]))))
positionZone = header["zone"]
positionItem = header["item"]

data = data[38:]
surprisalIndices = sorted([y for x, y in header.items() if x.startswith("Surprisal")])
surprisals = []
for line in data:
   surprisals.append([float(line[x]) for x in surprisalIndices])
surprisals = torch.FloatTensor(surprisals)
print(surprisals.size())
  
import torch.optim
probabilitiesTotal = torch.exp(-surprisals)

import torch.nn

interpolatedSurprisals = []
import numpy

losses = []

context = 37
for j in range(2, context):
  probabilities = probabilitiesTotal[:,:j]
  interpolationLogits = torch.zeros(j, requires_grad=True) #, require_grad=True)
  optim = torch.optim.SGD([interpolationLogits], lr=1.0)

  gradientSquaredNormHasBeenSmall = 0
  for i in range(10000000):
     interpolatedProbabilities = torch.matmul(probabilities, torch.nn.Softmax()(interpolationLogits))
     #print(interpolatedProbabilities.size())
   
     logLoss = -torch.log(interpolatedProbabilities).mean() + 0.001 * (interpolationLogits * interpolationLogits).mean()
     optim.zero_grad()
     logLoss.backward()
     gradientSquaredNorm = (interpolationLogits.grad.data * interpolationLogits.grad.data).sum().data.numpy()

     if gradientSquaredNorm < 1e-12:
        gradientSquaredNormHasBeenSmall += 1
     else:
        gradientSquaredNormHasBeenSmall = 0
     if i % 100 == 1:
        print(torch.nn.Softmax()(interpolationLogits))
        print(logLoss.data.numpy())
        print(gradientSquaredNorm)
        print(losses)
     if gradientSquaredNormHasBeenSmall > 500:
       losses.append(numpy.asscalar(logLoss.data.numpy()))
       break

     optim.step()
 #    if logLoss.data.numpy() <= 4.778:
#       break
   
  print(torch.nn.Softmax()(interpolationLogits))
  interpolatedSurprisals.append(-torch.log(interpolatedProbabilities).data.numpy())

print(len(interpolatedSurprisals))


with open(NATURALSTORIES_PATH+"/merged"+infix+"-linear-interpolated.txt", "w") as outFile:
  print("\t".join(["zone", "item"] + ["Surprisal_LinearInter_"+str(j-1) for j in range(2, context)]), file=outFile)
  for index, line in zip(range(len(data)), data):
     print("\t".join([line[positionZone], line[positionItem]] + [str(interpolatedSurprisals[j][index]) for j in range(0, context-2)]), file=outFile)

