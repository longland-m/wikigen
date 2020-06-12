# Functions to do the greedy similarity maximisation for article:node assignments
# All code is original


import random


def computeSimSum(G, similarityMatrix, asgn):
  """ Compute the total similarity sum for the current node:article assignment """
  S = sum([similarityMatrix[asgn[j], asgn[i]] 
          for j in range(len(G)) for i in list(G[j])])
  return S    
      

def greedySimilarityMax(G, similarityMatrix, nrounds=5):
  
  pairList = [(a,b) for a in range(len(G)) for b in range(a)]
  maxSimSums = []
  asgns = []
  
  for i in range(nrounds):
  
    # get random indices for initial node:article assignment
    init_ids = list(range(len(G)))
    random.shuffle(init_ids)
    
    # assign articles to nodes and compute initial similarity sum
    curAsgn = dict((key, init_ids[key]) for key in range(len(G)))
    curSimSum = computeSimSum(G, similarityMatrix, curAsgn)
    
    # maximisation loop - repeats until S can't increase 
    while True:
      # for each node pair, swap the nodes recompute similarity sum
      simSums = []    
      for edge in pairList:
        tempAsgn = dict(curAsgn)
        tempAsgn[edge[0]] = curAsgn[edge[1]]
        tempAsgn[edge[1]] = curAsgn[edge[0]]
        # Recompute similarity sum
        tempSimSum = computeSimSum(G, similarityMatrix, tempAsgn)       
        simSums.append(tempSimSum)
      
      # find the max possible new similarity score 
      # then update curAsgn if the new max score > old score
      maxNewSimSum = max(simSums)
      if maxNewSimSum > curSimSum:     
        nodesToSwap = pairList[simSums.index(maxNewSimSum)]
        oldAsgn = dict(curAsgn)
        curAsgn[nodesToSwap[0]] = oldAsgn[nodesToSwap[1]]
        curAsgn[nodesToSwap[1]] = oldAsgn[nodesToSwap[0]]      
        curSimSum = maxNewSimSum # no need to recompute, know the value already  
      else:
        break
    
    maxSimSums.append(curSimSum)
    asgns.append(curAsgn)
  
  bestRound = maxSimSums.index(max(maxSimSums))
  bestAsgn = asgns[bestRound]
  print('Best S = ' + str(maxSimSums[bestRound]))
  
  return bestAsgn
      
      
