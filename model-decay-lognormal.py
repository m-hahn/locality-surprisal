#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

import torch
from torch.autograd import Variable
import math


from pyro.infer import SVI
from pyro.optim import Adam


import pyro
import pyro.distributions as dist

import pyro
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI
from pyro.optim import Adam



print("Reading data")
with open("/u/scr/mhahn/langmod/naturalstories/dataZ.tsv", "r") as inFile:
#  data = [next(inFile).strip().split("\t") for _ in range(1000)] #x in inFile.read().strip().split("\n")]
  data = [x.split("\t") for x in inFile.read().strip().split("\n")]
  print("Read data")
  header = dict(zip([x[1:-1] for x in data[0]], xrange(len(data))))
print(header)
data = data[1:]



for i in range(len(header)):
   if not data[0][i].startswith('"'):
      for d in data:
        d[i] = float(d[i])
   else:
      for d in data:
        d[i] = d[i][1:-1]

print("Processed")

from math import log, exp
from random import shuffle



N = len(data)
M = int(max([line[header["tokenID.Renumbered"]] for line in data]))
L = int(max([line[header["WorkerId.Renumbered"]] for line in data]))

#print(M)
#print(L)
#quit()


alpha_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))
beta1_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))
decay_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))
kappa_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))

log_sigma_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))





byParticipantIntercept_Variance_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))
byItemIntercept_Variance_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))

byParticipantSlope_Variance_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))
byItemSlope_Variance_Prior = Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0])))

softplus = torch.nn.Softplus()

def logExp(x):
   return torch.log(1+torch.exp(x))


#
#def normal_product(loc, scale):
#    z1 = pyro.sample("z1", dist.Normal(loc, scale))
#    z2 = pyro.sample("z2", dist.Normal(loc, scale))
#    y = z1 * z2
#    return y
#
#def make_normal_normal():
#    mu_latent = pyro.sample("mu_latent", dist.Normal(0, 1))
#    fn = lambda scale: normal_product(mu_latent, scale)
#    return fn
#
#print(make_normal_normal()(1.))
#
#decay = pyro.sample("decay", dist.Normal(0.0, 10.0))
#print(decay)
#print(decay+1)
#print(decay*2)
#
#quit()

def model(data):
#  print("in model")
#  decay = pyro.sample("decay", dist.Normal(Variable(torch.FloatTensor([0.0])), Variable(torch.FloatTensor([10.0]))))
  decay = pyro.sample("decay", dist.Normal(0.0, 10.0))

#  distances = torch.FloatTensor(xrange(1, 36))
#  print(decay)
#  print(decay+1)
#  print(decay*2)
#  quit()

  #print(torch.log(distances))
  distances = xrange(1,36)
  contributions = [torch.exp(decay * log(d)) for d in distances]
 # contributions = torch.exp(torch.mul(torch.log(distances), decay))
  
  byParticipant_Intercept_logVariance = pyro.sample("byParticipant_Intercept_logVariance", byParticipantIntercept_Variance_Prior)
  byItem_Intercept_logVariance = pyro.sample("byItem_Intercept_logVariance", byItemIntercept_Variance_Prior)
  byParticipant_SurprisalSlope_logVariance = pyro.sample("byParticipant_SurprisalSlope_logVariance", byParticipantSlope_Variance_Prior)
  byParticipant_LogWordFreqSlope_logVariance = pyro.sample("byParticipant_LogWordFreqSlope_logVariance", byParticipantSlope_Variance_Prior)

  byItem_Slope_logVariance = pyro.sample("byItem_Slope_logVariance", byItemSlope_Variance_Prior)


  alpha = pyro.sample("alpha", alpha_Prior)
#  print("alpha in model")
#  print(alpha)
  beta1 = pyro.sample("beta1", beta1_Prior)
  kappa = pyro.sample("kappa", kappa_Prior)
  log_sigma = pyro.sample("log_sigma", log_sigma_Prior)
  sigma = logExp(log_sigma)

  byParticipant_Intercept = pyro.sample("byParticipant_Intercept", Normal(torch.autograd.Variable(torch.zeros(L)), logExp(byParticipant_Intercept_logVariance) * torch.autograd.Variable(torch.ones(L))))
  byItem_Intercept = pyro.sample("byItem_Intercept", Normal(torch.autograd.Variable(torch.zeros(M)), logExp(byItem_Intercept_logVariance) * torch.autograd.Variable(torch.ones(M))))


  byParticipant_SurprisalSlope = pyro.sample("byParticipant_SurprisalSlope", Normal(torch.autograd.Variable(torch.zeros(L)), logExp(byParticipant_SurprisalSlope_logVariance) * torch.autograd.Variable(torch.ones(L))))

  byParticipant_LogWordFreqSlope = pyro.sample("byParticipant_LogWordFreqSlope", Normal(torch.autograd.Variable(torch.zeros(L)), logExp(byParticipant_LogWordFreqSlope_logVariance) * torch.autograd.Variable(torch.ones(L))))

  #print("Start of model")

  for q in pyro.irange("data_loop", len(data), subsample_size=50):
       point = data[q]

       participant = int(point[header["WorkerId.Renumbered"]])-1
       assert participant >= 0
       assert participant < L
       item = int(point[header["tokenID.Renumbered"]]) -1
       surprisals = torch.FloatTensor([point[header["Increment"+str(x)]] for x in range(0, 35)])
       effectiveSurprisal = sum([x*y for x, y in zip(surprisals , contributions)])
       mean = alpha
       mean = mean + byParticipant_Intercept[participant]
       mean = mean + byItem_Intercept[item]
       mean = mean + (beta1 + byParticipant_LogWordFreqSlope[participant]) * point[header["Surprisal0"]]
       mean = mean + (kappa + byParticipant_SurprisalSlope[participant]) * effectiveSurprisal

       pyro.sample("time_{}".format(q), dist.Normal(mean, sigma), obs=torch.FloatTensor([log(point[header["RT"]])]))
       if random.random() > 0.99:
           print(surprisals)
           print([mean, log(point[header["RT"]])])
#       print("...")
#       print(alpha)
#       print(byParticipant_Intercept[participant])
#       print(byItem_Intercept[item])
#       print(beta1)
#       print( byParticipant_LogWordFreqSlope[participant])
#       print("WORD")
#       print(point[header["word.x"]])
#       print(point[header["Surprisal0"]])
#       print((beta1 + byParticipant_LogWordFreqSlope[participant]) * point[header["Surprisal0"]])
#       print((kappa + byParticipant_SurprisalSlope[participant]))
#       print(effectiveSurprisal)
#       print(mean)
#
#
#       quit()
#       print(mean)
#       print(point[header["RT"]])
#       print(torch.FloatTensor([log(point[header["RT"]])]))
       #print("Sampled")




def guide(data):


  #print("in guide")
  mu_byParticipant_Intercept_logVariance = pyro.param("mu_byParticipant_Intercept_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_byParticipant_Intercept_logVariance = pyro.param("sigma_byParticipant_Intercept_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_byItem_Intercept_logVariance = pyro.param("mu_byItem_Intercept_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_byItem_Intercept_logVariance = pyro.param("sigma_byItem_Intercept_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_byParticipant_SurprisalSlope_logVariance = pyro.param("mu_byParticipant_SurprisalSlope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_byParticipant_SurprisalSlope_logVariance  = pyro.param("sigma_byParticipant_SurprisalSlope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))

  mu_byParticipant_LogWordFreqSlope_logVariance = pyro.param("mu_byParticipant_LogWordFreqSlope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_byParticipant_LogWordFreqSlope_logVariance  = pyro.param("sigma_byParticipant_LogWordFreqSlope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))

  mu_byItem_Slope_logVariance = pyro.param("mu_byItem_Slope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_byItem_Slope_logVariance = pyro.param("sigma_byItem_Slope_logVariance", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_alpha = pyro.param("mu_alpha", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_alpha = pyro.param("sigma_alpha", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_beta1 = pyro.param("mu_beta1", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_beta1 = pyro.param("sigma_beta1", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_kappa = pyro.param("mu_kappa", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_kappa = pyro.param("sigma_kappa", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_log_sigma = pyro.param("mu_log_sigma", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_log_sigma = pyro.param("sigma_log_sigma", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  mu_byParticipant_Intercept = pyro.param("mu_byParticipant_Intercept", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  sigma_byParticipant_Intercept = pyro.param("sigma_byParticipant_Intercept", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  mu_byItem_Intercept = pyro.param("mu_byItem_Intercept", Variable(torch.FloatTensor([0.0]*M), requires_grad=True))
  sigma_byItem_Intercept = pyro.param("sigma_byItem_Intercept", Variable(torch.FloatTensor([0.0]*M), requires_grad=True))
  mu_byParticipant_Intercept = pyro.param("mu_byParticipant_Intercept", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  sigma_byParticipant_Intercept = pyro.param("sigma_byParticipant_Intercept", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  mu_byItem_Intercept = pyro.param("mu_byItem_Intercept", Variable(torch.FloatTensor([0.0]*M), requires_grad=True))
  #print(M)
  #print(mu_byItem_Intercept)
  sigma_byItem_Intercept = pyro.param("sigma_byItem_Intercept", Variable(torch.FloatTensor([0.0]*M), requires_grad=True))
  mu_byParticipant_SurprisalSlope = pyro.param("mu_byParticipant_SurprisalSlope", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  sigma_byParticipant_SurprisalSlope = pyro.param("sigma_byParticipant_SurprisalSlope", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  mu_byParticipant_LogWordFreqSlope = pyro.param("mu_byParticipant_LogWordFreqSlope", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))
  sigma_byParticipant_LogWordFreqSlope = pyro.param("sigma_byParticipant_LogWordFreqSlope", Variable(torch.FloatTensor([0.0]*L), requires_grad=True))





  byParticipant_Intercept_logVariance = pyro.sample("byParticipant_Intercept_logVariance", dist.Normal(mu_byParticipant_Intercept_logVariance, logExp(sigma_byParticipant_Intercept_logVariance)))
  byItem_Intercept_logVariance = pyro.sample("byItem_Intercept_logVariance",  dist.Normal(mu_byItem_Intercept_logVariance, logExp(sigma_byItem_Intercept_logVariance)))
  byParticipant_SurprisalSlope_logVariance = pyro.sample("byParticipant_SurprisalSlope_logVariance", dist.Normal(mu_byParticipant_SurprisalSlope_logVariance, logExp(sigma_byParticipant_SurprisalSlope_logVariance)))
  byParticipant_LogWordFreqSlope_logVariance = pyro.sample("byParticipant_LogWordFreqSlope_logVariance", dist.Normal(mu_byParticipant_LogWordFreqSlope_logVariance, logExp(sigma_byParticipant_LogWordFreqSlope_logVariance)))

  byItem_Slope_logVariance = pyro.sample("byItem_Slope_logVariance", dist.Normal(mu_byItem_Slope_logVariance, logExp(sigma_byItem_Slope_logVariance)))


  alpha = pyro.sample("alpha", dist.Normal(mu_alpha, logExp(sigma_alpha)))
#  print("Guide")
#  print(alpha)
#  print(mu_alpha)
#  print(sigma_alpha)
  beta1 = pyro.sample("beta1", dist.Normal(mu_beta1, logExp(sigma_beta1)))


  mu_decay = pyro.param("mu_decay", Variable(torch.FloatTensor([0.0]), requires_grad=True))
  sigma_decay = pyro.param("sigma_decay", Variable(torch.FloatTensor([0.0]), requires_grad=True))

  
  decay = pyro.sample("decay", dist.Normal(mu_decay, logExp(sigma_decay)))

  kappa = pyro.sample("kappa", dist.Normal(mu_kappa, logExp(sigma_kappa)))
  log_sigma = pyro.sample("log_sigma", dist.Normal(mu_log_sigma, logExp(sigma_log_sigma)))

  byParticipant_Intercept = pyro.sample("byParticipant_Intercept", dist.Normal(mu_byParticipant_Intercept, logExp(sigma_byParticipant_Intercept)))
  byItem_Intercept = pyro.sample("byItem_Intercept", dist.Normal(mu_byItem_Intercept, logExp(sigma_byItem_Intercept)))
#  print(mu_byItem_Intercept)
#  print(sigma_byItem_Intercept)
#  print(byItem_Intercept)


  byParticipant_SurprisalSlope = pyro.sample("byParticipant_SurprisalSlope", dist.Normal(mu_byParticipant_SurprisalSlope, logExp(sigma_byParticipant_SurprisalSlope)))
  byParticipant_LogWordFreqSlope = pyro.sample("byParticipant_LogWordFreqSlope", dist.Normal(mu_byParticipant_LogWordFreqSlope, logExp(sigma_byParticipant_LogWordFreqSlope)))





       

adam_params = {"lr": 0.001, "betas": (0.90, 0.999)} # , "eps" : 1e-5
optimizer = Adam(adam_params)


pyro.clear_param_store()

from pyro.infer import  Trace_ELBO
# setup the inference algorithm
svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO()) #, num_particles=7)

n_steps = 40000000
# do gradient steps
for step in range(1,n_steps):
#    print("DOING A STEP")
#    print(".......")
#    print(step)

 #   print 
#    quit()

    svi.step(data)

    if step % 50 == 0:
      print(".....")
      print(step)
      for name in pyro.get_param_store().get_all_param_names():
#         if "decay" not in name and "alpha" not in name and "beta" not in name and "kappa" not in name:
#           continue
         print [name, pyro.param(name).data.numpy()]
         if (pyro.param(name).data != pyro.param(name).data).any():
            print("ISNA")
            quit() 
    



