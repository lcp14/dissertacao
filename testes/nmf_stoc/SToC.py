# -*- coding: iso-8859-15 -*-
from sklearn.preprocessing import normalize
import numpy as np
import sys
from scipy import sparse

def getIrredutibleMatrix(W,H,n_topics):
	#Grafo Tripartite W e H normalizados
	W_normalized_line = normalize(W,axis=1, norm="l1")
	H_normalized_line = normalize(H,axis=1, norm="l1")
	W_normalized_col = normalize(W,axis=0, norm="l1")
	H_normalized_col = normalize(H,axis=0, norm="l1")

	#Topic Trasition Graph
	topXtop = [[0.0 for x in range(n_topics)] for y in range(n_topics)]

	#print H_normalized_col[i][0]*H_normalized_line[i][0]
	for y in range(0, n_topics):
		for k in range(0,n_topics):
			soma = 0
			for i in range(len(H_normalized_line)):
				soma = soma + H_normalized_col[i][y]*H_normalized_line[i][k]
			for j in range(len(W_normalized_line)):
				soma = soma + W_normalized_col[j][y]*W_normalized_line[j][k]
			topXtop[y][k] = soma
	


	#matrix M
	topXtop_norm = normalize(topXtop,axis=1, norm="l1")

	damp_factor = 0.95

	diagonal = 0
	for i in range(n_topics):
		for j in range(n_topics):
			if i == j:
				diagonal = diagonal + topXtop_norm[i][j] 

	restartProbability = diagonal / n_topics

	minConn = (1 - restartProbability) * 1/(n_topics-1)

	#print minConn

	indiceee = 0
	for i in range(n_topics):
		for j in range(n_topics):
			if(topXtop_norm[i][j] <= minConn):
				topXtop_norm[i][j] = 0
			
			if(sum(topXtop_norm[i]) <= topXtop_norm[i][i] ):
				for j in range(n_topics):
					if(i != j):
						topXtop_norm[i][j] = (1 - topXtop_norm[i][i]) * 1/(n_topics)
			else:
				norm = sum(topXtop_norm[i])
				for j in range(n_topics):
					topXtop_norm[i][j] = topXtop_norm[i][j]/norm
	
	irredutivel = damp_factor * topXtop_norm + (((1.0 - damp_factor) * (1.0 / n_topics)) * np.ones((n_topics,n_topics)))

	return irredutivel



def getDiameter(n_topics,randomWalkMatrix):
	
	diagonal = 0
	for i in range(n_topics):
		for j in range(n_topics):
			if i == j:
				diagonal = diagonal + randomWalkMatrix[i][j] 

	restartProbability = diagonal / n_topics

	minConn = (1 - restartProbability) * 1/(n_topics-1)

	fullMatriz = np.zeros((n_topics,n_topics))

	for i in range(n_topics):
		for j in range(n_topics):
			if randomWalkMatrix[i][j] > minConn :
				fullMatriz[i][j] = 1

	distances = sparse.csgraph.dijkstra(fullMatriz)

	diameter = 1
	for i in range(n_topics):
		for j in range(n_topics):
			if ((distances[i][j] != np.inf) & (distances[i][j] > diameter)):
				diameter = distances[i][j]


	return int(diameter)

def getNumLinks(k,matrix):
	numLinks = 0 
	
	diagonal = 0
	for i in range(k):
		for j in range(k):
			if i == j:
				diagonal = diagonal + matrix[i][j] 

	restartProbability = diagonal / k
	minConn = (1 - restartProbability) * 1/(k-1)

	for i in range(k):
		for j in range(k):
			if((i != j) & (matrix[i][j] > minConn)):
				numLinks = numLinks + 1

	meanNumLinks = numLinks / k

	return meanNumLinks

def selectBestTopicPair(numTopics, matrix, topicIndexes, clusterAssignments, mapTopics, originalNumTopics,alpha,minProbability,maxCohesion,transitionMatrix, topicSize):
	
	selectedRows = np.zeros(numTopics)
	selectedColumns = np.zeros(numTopics)
	maxProbabilities = np.zeros(numTopics)

	matrixSize = numTopics * numTopics

	dirtyTopics = np.zeros(numTopics)
	temporaryMatrix = np.zeros((numTopics,numTopics))

	#print "origi" , originalNumTopics
	#print "numTopic", numTopics

	for i in range(numTopics):
		topic1 = topicIndexes[i]
		#print "Topic 1 " , topic1
		if (topic1 >= originalNumTopics):
			topic1 = mapTopics[topic1 - originalNumTopics] 

		for j in range(i+1,numTopics):
			topic2 = topicIndexes[j]
			if(topic2 >= originalNumTopics):
				topic2 = mapTopics[topic2 - originalNumTopics]

			resultingCluster = clusterAssignments[int(topic1)] + clusterAssignments[int(topic2)]
			numFactors = sum(resultingCluster)
			#print numFactors
			
			#Original - > 15 topicos 11x1 2x2 1x3 1xN
			#diagResultingCluster = np.diag(resultingCluster)
			#resultingCluster = np.asmatrix(resultingCluster)
			#auxiliaryMatrix = resultingCluster.transpose() * resultingCluster
			#auxiliaryMatrix = auxiliaryMatrix - diagResultingCluster
			#print "aux" ,auxiliaryMatrix
			#print clusterAssignments
			#resultingVector = np.diag(transitionMatrix * auxiliaryMatrix.T)
			#print resultingVector
			#maxObserverdCohesion = max(resultingVector) / (numFactors - 1)
			#print maxObserverdCohesion
			#maxCohesion = 1.0 / numFactors
			#deltaCohesion = (maxObserverdCohesion - maxCohesion)/maxCohesion
			#transitionProbability = ((matrix[i][j] + matrix[j][i])/2)
			#deltaUnicity = (transitionProbability - minProbability)/minProbability
			#temporaryMatrix[i][j] = alpha*deltaCohesion + (1.0 - alpha)*deltaUnicity
			

			#Tentativa 1 -> 20 topicos ,13x1 3x2 1x3 1x4 1xN
			#diagResultingCluster = np.diag(resultingCluster)
			#resultingCluster = np.asmatrix(resultingCluster)
			#auxiliaryMatrix = resultingCluster.transpose() * resultingCluster
			#auxiliaryMatrix = auxiliaryMatrix - diagResultingCluster
			#resultingVector = np.diag(transitionMatrix * auxiliaryMatrix.T)
			#maxObserverdCohesion = max(resultingVector) / (numFactors - 1)
			#maxCohesion = 1.0 / numFactors
			#deltaCohesion = (maxObserverdCohesion - maxCohesion)/maxCohesion
			#merge = (1 + (1/numFactors))
			#transitionProbability = (np.sqrt(matrix[i][j] * matrix[j][i])) * merge
			#deltaUnicity = (transitionProbability - minProbability)/minProbability
			#temporaryMatrix[i][j] = alpha*deltaCohesion + (1.0 - alpha)*deltaUnicity


			#Tentativa 2 - > 1 topico
			#lineI = matrix[i]
			#columnJ = np.matrix(matrix).transpose()[j].getA()[0]
			#penality = (1.0 + (1.0 / numFactors ))
			#transitionProbability = np.sqrt(lineI.dot(columnJ.T)) * penality
			#temporaryMatrix[i][j] = ((transitionProbability - minProbability)/minProbability)

			
			#Tentativa 3 
			merge = (1.0 + (1.0/numFactors))
			dist1 = matrix[i]
			dist2 = matrix[j]
			#media1 = np.average(dist1)
			#media2 = np.average(dist2)
			score = 0
			for k in range(len(matrix[i])):
				score += np.sqrt(dist1[k]*dist2[k])
			#score = np.sqrt(1 - ( 1 / np.sqrt(media1*media2*len(matrix[i])*len(matrix[j]))) * score)
			cohesion = (score) * merge
			temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability

			
			#usado nos experimentos
			#tentativa 4 
			#merge = (1 + (1/numFactors))
			#dist1 = matrix[i]
			#dist2 = matrix[j]
			#score = np.sqrt(dist1.dot(dist2.T))
			#cohesion = (score) * merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability




			# tentativa 5
			#merge = (1 + (1/numFactors))
			#cohesion = (sqrt((matrix[i][j]+matrix[j][i])/2.0))*merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability

			# 1 topico
			#merge = (1 + (1/numFactors))
			#cohesion = -1.0 * np.log((np.sqrt(matrix[i][j]*matrix[j][i])))*merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability

			## 1 topico
			#merge = (1 + (1/numFactors))
			#cohesion = -1.0 * np.log((np.sqrt(matrix[i].dot(np.matrix(matrix).transpose()[j].getA()[0]))))*merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability

			## 1 topico
			#merge = (1 + (1/numFactors))
			#cohesion = -1.0 * np.log((np.sqrt(matrix[i].dot(matrix[j].T))))*merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability



			#Paper -> 9 topicos
			#merge = (1 + (1/numFactors))
			#cohesion = (sqrt(matrix[i][j]*matrix[j][i]))*merge
			#temporaryMatrix[i][j] = (cohesion-minProbability)/minProbability

			
	#print temporaryMatrix

	probabilities = np.reshape(temporaryMatrix.T,matrixSize,1)

	X = sorted(enumerate(probabilities), key=lambda x: x[1],reverse=True)
	indices, sortedProbabilities = list(map(list, list(zip(*X))))
	
	#print indices
	#print probabilities

	index = 0
	for i in range(matrixSize):
		if sortedProbabilities[i] != 0 :
			#selectedRow = np.floor(indices[i]/numTopics) + 1
			selectedRow = int(np.floor(indices[i]/numTopics))
			#print selectedRow
			selectedColumn = int(np.mod(indices[i],numTopics))
			#print selectedColumn

			if ((dirtyTopics[selectedRow] != 1) & (dirtyTopics[selectedColumn] != 1)):
				selectedRows[index] = topicIndexes[selectedRow]
				selectedColumns[index] = topicIndexes[selectedColumn]
				maxProbabilities[index] = sortedProbabilities[i]
				index = index + 1

				dirtyTopics[selectedColumn] = 1
				dirtyTopics[selectedRow] = 1

			#input("ok")

	return selectedRows, selectedColumns, maxProbabilities


def getStationaryDistribution(dampFactor, epsilon, numTopics, transitionMatrix):
	#não é necessario
	#numTopics = max(len(transitionMatrix))

	vectorOfOnes = np.ones(numTopics)
	residualVector = []
	residual = 1
	rankVector = 1/numTopics * vectorOfOnes

	#page rank iterativo
	while residual >= epsilon:
		prevRankVector = rankVector
		auxiliaryVector = dampFactor * (np.dot(rankVector,transitionMatrix))
		beta = 1 - np.linalg.norm((auxiliaryVector), ord=1)
		rankVector = auxiliaryVector + beta*1/numTopics * vectorOfOnes

		residual = np.linalg.norm((rankVector - prevRankVector),ord=1)

		residualVector.append(residual)

	return rankVector

def updateClustersInfo(selectedTopic1, selectedTopic2, initialNumTopics, newTopicId, mapTopics, clusterAssignments):
	if(selectedTopic1 > initialNumTopics):
		selectedTopic1 = mapTopics[selectedTopic1 - initialNumTopics - 1]

	if(selectedTopic2 > initialNumTopics):
		selectedTopic2 = mapTopics[selectedTopic2 - initialNumTopics - 1]

	if selectedTopic1 > selectedTopic2:
		mapTopics[newTopicId - initialNumTopics - 1] = selectedTopic1
		clusterAssignments[selectedTopic2] = clusterAssignments[selectedTopic2] + clusterAssignments[selectedTopic1] 
	else:
		mapTopics[newTopicId - initialNumTopics - 1] = selectedTopic2
		clusterAssignments[selectedTopic1] = clusterAssignments[selectedTopic1] + clusterAssignments[selectedTopic2]

	return mapTopics, clusterAssignments

#M = irredutile
def joinTopics(k,irredutible):
	np.set_printoptions(threshold=sys.maxsize)

	#print "Irredutivel"
	#print irredutible
	t = {}
	for i in range(k):
		t[i] = [i]

	epsilon = 0.00001
	alpha = 0.5
	dampFactor = 0.95
	numTopics = k

	finalNumTopics = k

	topicIndexes = []
	mapIndexTopic = []

	for i in range(k):
		topicIndexes.append(i)
		mapIndexTopic.append(i)

	#for i in range(k):
		#topicIndexes.append(-1)
	#	mapIndexTopic.append(-1)

	#print topicIndexes
	compose = [[x] for x in range(k)]

	randomWalkMatrix = irredutible

	numHops = getDiameter(k, randomWalkMatrix)
	for i in range(0,numHops-1):
		randomWalkMatrix = np.dot(randomWalkMatrix, irredutible)

	transitionMatrix = randomWalkMatrix


	diagonal = sum(np.diag(irredutible))
	leavingProbability = (1 - diagonal/k)
	maxCohesion = 1 - leavingProbability

	meanNumLinks = getNumLinks(k, irredutible)
	minProbability = leavingProbability * 1/(meanNumLinks)

	topicSize = np.ones((2*k, 1))
	originalNumTopics = k
	mapTopics = np.zeros((k,1))
	clusterAssignments = np.eye(k, dtype=int)
	Tsize = np.ones(k)
	
	#arq = open("experimento1p2","w")	
	t2 = []
	#ok ate aqui
	while numTopics > 1 :
		#print numTopics
		currentIndex = 0

		selectedRows, selectedColumns, maxProbabilities = selectBestTopicPair(numTopics, randomWalkMatrix, topicIndexes, clusterAssignments, mapTopics, originalNumTopics, alpha, minProbability, maxCohesion, transitionMatrix,topicSize)

		while ((currentIndex < numTopics) & (maxProbabilities[currentIndex] > 0)):
		#if ((currentIndex < numTopics) & (maxProbabilities[currentIndex] > 0)):
			selectedRow = mapIndexTopic[int(selectedRows[currentIndex])]
			selectedColumn = mapIndexTopic[int(selectedColumns[currentIndex])]
			#print numTopics
			#print((finalNumTopics, topicIndexes[selectedRow], topicIndexes[selectedColumn], maxProbabilities[currentIndex]))
			t2.append((finalNumTopics, topicIndexes[selectedRow], topicIndexes[selectedColumn], maxProbabilities[currentIndex]))
			#arq.write(str((finalNumTopics, topicIndexes[selectedRow], topicIndexes[selectedColumn], maxProbabilities[currentIndex])))


			#print topicSize
			#input("ok")
			t[finalNumTopics] = [topicIndexes[selectedRow]] + [topicIndexes[selectedColumn]]
			#print selectedColumn
			
			stationayProbabilities = getStationaryDistribution(dampFactor, epsilon, numTopics, irredutible)

			#print "Topic Index Row: ", topicIndexes[selectedRow], " Topic Index Column: ", topicIndexes[selectedColumn], " Max Probabilities: ", maxProbabilities[currentIndex], " Min Probability: ", minProbability, " LeavingProbability: ", leavingProbability 

			#empurra a matriz pra frente, deixando os dois ultimos
			for i in range(numTopics-2):
				
				diff = 0
				index = topicIndexes[i]
				if index >= topicIndexes[selectedRow]:
					diff = diff + 1
				if index >= topicIndexes[selectedColumn]:
					diff = diff + 1
				#print diff
				index = i + diff
				while index < numTopics:
					if ((topicIndexes[index] != topicIndexes[selectedRow]) & (topicIndexes[index] != topicIndexes[selectedColumn])):
						break
					index = index + 1

				if index <= numTopics:
					topicIndexes[i] = topicIndexes[index]
					mapIndexTopic[topicIndexes[index]] = i

			
			finalNumTopics = finalNumTopics + 1

			topicIndexes[numTopics - 2] = finalNumTopics -1

			topicSize[topicIndexes[numTopics-2]] = topicSize[topicIndexes[selectedRow]] + topicSize[topicIndexes[selectedColumn]]
			
			if len(mapIndexTopic) < finalNumTopics:
				diff = finalNumTopics - len(mapIndexTopic) 
				for i in range(diff):
					mapIndexTopic.append(-1)
				

			mapIndexTopic[finalNumTopics - 1] = numTopics - 2

			#P4

			mapTopics, clusterAssignments = updateClustersInfo(selectedRow, selectedColumn, originalNumTopics, finalNumTopics, mapTopics, clusterAssignments)

			

			newMatrix = np.zeros((numTopics-1,numTopics-1))
			mapNewOldIndex = np.zeros(numTopics-1)

			validRowIndex = -1
			for i in range(numTopics):
				if((i != selectedRow) & (i != selectedColumn)):
					validRowIndex = validRowIndex + 1
					mapNewOldIndex[validRowIndex] = i
					validColumnIndex = validRowIndex - 1
					for j in range(i,numTopics):
						if((j != selectedRow) & (j != selectedColumn)):
							validColumnIndex = validColumnIndex + 1
							newMatrix[validRowIndex,validColumnIndex] = irredutible[i,j]
							newMatrix[validColumnIndex,validRowIndex] = irredutible[j,i]

			#print newMatrix[0]

			numTopics = numTopics - 1

			for i in range(numTopics-1):
				
			
				newMatrix[i][numTopics - 1] = irredutible[int(mapNewOldIndex[i])][selectedRow] + irredutible[int(mapNewOldIndex[i])][selectedColumn]
				
				newMatrix[numTopics - 1][i] = irredutible[selectedRow][int(mapNewOldIndex[i])] * stationayProbabilities[selectedRow]/(stationayProbabilities[selectedRow] + stationayProbabilities[selectedColumn]) + irredutible[selectedColumn][int(mapNewOldIndex[i])] * stationayProbabilities[selectedColumn]/(stationayProbabilities[selectedColumn] + stationayProbabilities[selectedRow])

			newMatrix[numTopics-1][numTopics-1] = (irredutible[selectedRow,selectedRow] + irredutible[selectedRow,selectedColumn]) * stationayProbabilities[selectedRow]/(stationayProbabilities[selectedRow] + stationayProbabilities[selectedColumn]) + (irredutible[selectedColumn,selectedColumn] + irredutible[selectedColumn,selectedRow]) * stationayProbabilities[selectedColumn]/(stationayProbabilities[selectedRow] + stationayProbabilities[selectedColumn])
			
			for i in range(numTopics):
				for j in range(numTopics):
					if(np.isnan(newMatrix[i,j])):
						newMatrix[i,j] = 0
			irredutible = newMatrix

			currentIndex = currentIndex + 1

		randomWalkMatrix = irredutible

		numHops = getDiameter(numTopics, randomWalkMatrix)
		
		for i in range(0,numHops-1):
			randomWalkMatrix = np.dot(randomWalkMatrix, irredutible)

		diagonal = sum(np.diag(irredutible))
		leavingProbability = (1 - diagonal/numTopics)

		meanNumLinks = getNumLinks(numTopics, irredutible)
		minProbability = leavingProbability * 1/(meanNumLinks)

		if currentIndex == 0 :
			break

	print(("NumTopics: ", numTopics))

	#arq.close()

	#print clusterAssignments
	return t,t2


