import random
import numpy as np
from scipy.stats import lognorm

class NC:

  def __init__(self, nSeg_, nObs_, link_times):
    self.nSeg = nSeg_
    self.nObs = nObs_
        #nObs = kEnd - kBeg + 1
    self.lTimes = link_times
    self.tTimes = None
    self.tTotals = None
    self.output = None
    self.zInfo = {}
    self.aVal = None
    self.bVal = None
    self.cVal = None
    self.zBest = None
    self.ksMaxD = None
    self.ksCritD = None
    self.ksPass = None
    self.tPos = np.empty(self.nObs, dtype=float)
    self.tNeg = np.empty(self.nObs, dtype=float)
    self.tConv = np.empty(self.nObs, dtype=float)

  def get_means_stdv(self):
    """returns average and stdv of a list of values (timelist)"""
    means = []
    stdvs = []
    for segment in sorted(self.lTimes.keys()):
      sum_ = 0
      var = 0.
      for times in self.lTimes[segment]:
        sum_ = sum_ + times
      average = sum_/len(self.lTimes[segment])
      means.append(average)

      for times in self.lTimes[segment]:
        var = var + (times-average)**2
      var = var/len(self.lTimes[segment])
      std = np.sqrt(var)
      stdvs.append[std]

    return means, stdvs


  def gen_obs(self):
    uMean = uStdDev = cMean = pUnc = np.empty(self.nSeg, dtype=float)
    sProb = cProb = np.array([np.empty(4, dtype=float) for i in xrange(self.nSeg)])
    jRV = sVeh = tRte = np.empty(self.nObs, dtype=float)
    tVeh = np.array([np.empty(self.nObs, dtype=float) for i in xrange(self.nSeg)])
    times = [[] for i in xrange(self.nSeg)]
    totals = []

    uMean, uStdDev = self.get_means_stdv()
#Randomize

    # for k in xrange(1,nSeg+1):
      # 'Read the means and standard deviations for uncongested and congested times'
      # uMean(k) = Worksheets("Route").Cells(3, 3 + k).Value
      # uStdDev(k) = Worksheets("Route").Cells(4, 3 + k).Value
      # cMean(k) = Worksheets("Route").Cells(5, 3 + k).Value
      # cStdDev(k) = Worksheets("Route").Cells(6, 3 + k).Value
      # 'Read the state transitions'
      # sProb(k, 1) = Worksheets("Route").Cells(8, 3 + k).Value
      # cProb(k, 1) = sProb(k, 1)
      # sProb(k, 2) = Worksheets("Route").Cells(9, 3 + k).Value
      # cProb(k, 2) = cProb(k, 1) + sProb(k, 2)
      # sProb(k, 3) = Worksheets("Route").Cells(10, 3 + k).Value
      # cProb(k, 3) = cProb(k, 2) + sProb(k, 3)
      # sProb(k, 4) = Worksheets("Route").Cells(11, 3 + k).Value
      # cProb(k, 4) = cProb(k, 3) + sProb(k, 4)
      # pUnc(k) = cProb(k, 2)


#Initialize the states of the vehicles'
    for j in xrange(self.nObs)
      k = 1 #'use the state probabilities for the first segment'
      srv = random.uniform(0,1) #'state selection random variable'
      if srv < cProb[0][k]):
        sVeh[j] = 1
      elif srv < cProb[1][k]:
        sVeh[j] = 2
      elif srv < cProb[2][k]:
        sVeh[j] = 3
      else:
        sVeh[j] = 4
  
    for j in xrange(self.nObs):
      rvu = random.uniform(0,1) #'uncongested travel rate - driver type'
      tTot = 0.
      for k in xrange(self.nSeg)
        if sVeh[j] <= 2: 
          rvt = rvu
          mean = uMean[k]
          StdDev = uStdDev[k]
        else:
          rvt = random.uniform(0,1)
          mean = cMean[k]
          StdDev = cStdDev[k]

        tSeg = lognorm.ppf(rvt, s = StdDev, scale = np.exp(mean))
        times[k].append(tSeg)
        tVeh[k][j] = tSeg
        tTot = tTot + tSeg
        #'Update the state of the vehicle'
        if k < self.nSeg: 
          if sVeh[j] == 1 or sVeh[j] == 3:
            srv = 0.5 * random.uniform(0,1) #'state selection random variable'
            if srv < cProb[0][k+1]:
              sVeh[j] = 1 
            else:
             sVeh[j] = 2
          else:# '(sVeh(j) = 2) Or (sVeh(j) = 4)'
            srv = 0.5 + 0.5 * random.uniform(0,1) #'state selection random variable'
            if srv < cProb[2][k+1]:
              sVeh[j] = 3
            else:
              sVeh[j] = 4
      totals.append(tTot)

    self.tTimes = times
    self.tTotals = totals

  def analyze_route():
   # 'Set Parameter Values'
    jTn = round(10000. / self.nObs, 0)
    nSamp = jTn * self.nObs
    nKS = 200
    ksCritD = 1.36 * (2 * self.nObs / (self.nObs**2))**0.5

    #'Dimension the arrays'
    tSamp = np.empty(nSamp, dtype=float)
    tRte = tEst = tTemp = np.empty(self.nObs, dtype=float)
    tVeh = self.tTimes

   # Worksheets("RouteEval").Select
    #Range("A1").Select

   # 'Sort the sampled segment travel time arrays into ascending order'
  #'In this case, nObs is just an index number, not the number of the vehicle'
    for k in xrange(self.nSeg):
      for j in xrange(self.nObs):
        tTemp[j] = tVeh[k][j]
      self.val_sort(tTemp)
      tVeh[k] = tTemp
    tVeh[0][0] = tVeh[0][1]
    #tRte(0) = tRte(1)
    OP = [[i+1 for i in xrange(self.nObs)]]
    tRte = [el for el in self.tTotals]
    OP.append(tRte)
    for k in xrange(self.nSeg):
      OP.append(tVeh[k])
        
    #'Compute the positive and negative distributions'
    for j in xrange(self.nObs):
      self.tPos[j] = 0
      self.tNeg[j] = 0
      for k in xrange(self.nSeg):
        self.tPos[j] = self.tPos[j] + tVeh[k][j]
        if k%2 == 0:
          self.tNeg[j] = self.tNeg[j] + tVeh[k][j]
        else:
          self.tNeg[j] = self.tNeg[j] + tVeh[k][self.nObs - j]

    #' Generate convolution-based sample travel times'
    for n in xrange(nSamp):
      tSamp[n] = 0.
      for k in xrange(self.nSeg):
        nVal = int(max(0, min(self.nObs-1 * random.uniform(0,1), self.nObs-1)))
        tSamp[n] = tSamp[n] + tVeh[k][nVal])

    #' Generate tConv'
    self.val_sort(tSamp)
    for j in xrange(self.nObs)
      self.tConv[j] = tSamp[j * jTn]

    #'Sort the positive, negative, and uncorrelated synthesized distributions'
    self.val_sort(self.tPos)
    self.val_sort(self.tNeg)
    self.val_sort(self.tConv)
    OP.append(self.tPos)
    OP.append(self.tNeg)
    OP.append(self.tConv)
   # 'Find the best composite distribution'
    aBest = 0.
    bBest = 0.
    cBest = 0.
    zBest = 1e+30

    for na in np.linspace(0,100,21):
      a = round(0.01 * na, 2)
      for nb in np.linspace(0,100 - na, (100-na)/5 + 1)
        b = round(0.01 * nb, 2)
        aPb = round(a + b, 2)
        c = round(1 - aPb, 2)
        if c < 0: 
          c = 0
        if c > 1:
          c = 1
        zTest = 0
      
      #'Create the proportionally sampled distribution'
      for n in xrange(nSamp)
        rVar = random.uniform(0,1)
        rj = int(self.nObs * random.uniform(0,1))
        if rVar < a:
          tSamp[n] = self.tPos[rj]
        elif rVar >= a and rVar < aPb:
          tSamp[n] = self.tNeg[rj]
        elif rVar >= aPb: 
          tSamp[n] = self.tConv[rj]

      #' Compute tEst'
      self.val_sort(tSamp)
      for j in xrange(self.nObs)
        tEst[j] = tSamp[jTn * j]
      
      self.eval_ks(zTest, ksMaxD, tRte, tEst, nKS)
     
      if zTest < zBest:
        zBest = zTest
        aBest = a
        bBest = b
        cBest = c
        c = cBest

      m = m + 1
      zInfo['a'] = a
      zInfo['b'] = b
      zInfo['c'] = c
      zInfo['zTest'] = zTest
      

    #'Record the best values'
    a = round(aBest, 2)
    b = round(bBest, 2)
    c = round(cBest, 2)
    aPb = round(a + b, 2)

    #'Create a proportionally sampled distribution for the final a,b,c values'
    for n in xrange(nSamp):
      rVar = random.uniform(0,1)
      rj = self.nObs * random.uniform(0,1)
      if rVar < a:
        tSamp[n] = self.tPos[rj]
      elif rVar >= a and rVar < aPb:
        tSamp[n] = self.tNeg[rj]
      else: #'rVar >= aPb'
        tSamp[n] = self.tConv[rj]

    #' Compute tEst'
    self.val_sort(tSamp)
    for j in xrange(self.nObs):
      tEst[j] = tSamp[jTn * j]

    #'Evaluate the metrics'
    self.eval_ks(zTest, ksMaxD, tRte, tEst, nKS)
    if ksMaxD > ksCritD: 
      ksPass = 0 
    else: 
      ksPass = 1

    #'Record the results'
    self.aVal = a
    self.bVal = b
    self.cVal = c
    self.zBest = zTest
    self.ksMaxD = ksMaxD
    self.ksCritD = ksCritD
    self.ksPass = ksPass

    #'Add estimated values to output'
    OP.append(tEst)
    self.output = OP

  def eval_ks(zTest, ksMaxD, tRte, tEst, nKS):
    pVal = [[] for i in range(len(nKS))]
    tMin = min(tRte[0], tEst[0])
    tMax = max(tRte[self.nObs], tEst[self.nObs])
    dtKS = (tMax - tMin) / CSng[nKS]
    dp = 1. / CSng[self.nObs]

    for k in xrange(1,3):
      j = 1
      pVal[0][k] = 0
      for n in xrange(1,nKS + 1):
        tKS = dtKS * n + tMin
        while True:
          if j >= self.nObs: 
            pVal[n][k] = 1
            break
          if k == 1:
            if tRte[j] >= tKS:
              denom = tRte[j] - tRte[j - 1]
              if denom > 0: 
                pVal(k, n) = dp * (j - 1) + dp * (tKS - tRte[j - 1]) / denom
              break
            else:
              j = j + 1
          else:
            if tEst[j] >= tKS: 
              denom = tEst[j] - tEst[j - 1]
              if denom > 0:  
                pVal[n][k] = dp * (j - 1) + dp * (tKS - tEst[j - 1]) / denom
              break
            else:
              j = j + 1

      zTest = 0
      ksMaxD = 0
      for n in xrange(1, nKS +1)
        pDel = abs(pVal[n][2] - pVal[n][1])
        if pDel > ksMaxD:
          ksMaxD = pDel
        zTest = zTest + pDel*pDel

  def val_sort(self,tVals):
    #'Sorts the values into ascending order
    temp = []
    tValIdx = []

    for i in xrange(len(tVals)):
      temp.append(tVals[i])
      tValIdx.append(i)

    jump = len(tVals)/2

    while jump >= 1:
      while True:
        done = 1
        for i in xrange(len(tVals) - jump):
          j = i + jump
          if temp[i] > temp[j]: 
            Hold1 = temp[i]
            temp[i] = temp[j]
            temp[j] = Hold1
            Hold2 = tValIdx[i]
            tValIdx[i] = tValIdx[j]
            tValIdx[j] = Hold2
            done = 0
        if done == 1:
          break
      jump = int(jump/2)

    for i in xrange(len(tVals)):
      tVals[i] = temp[i]