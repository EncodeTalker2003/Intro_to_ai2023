# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import sqrt, log

from game import Agent
import os



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState, 0, self.depth)[1]
    
    def minimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maximize(gameState, depth, agentIndex)
        else:
            return self.minimize(gameState, depth, agentIndex)
    
    def maximize(self, gameState, depth, agentIndex): 
        next_agentIndex, next_depth = agentIndex + 1, depth
        if agentIndex == gameState.getNumAgents() - 1:
            next_agentIndex, next_depth = 0, depth - 1
        max_value, max_action = -float('inf'), Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            next_State = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxSearch(next_State, next_agentIndex, next_depth)[0]
            if value > max_value:
                max_value, max_action = value, action
        return max_value, max_action
    
    def minimize(self, gameState, depth, agentIndex):
        next_agentIndex, next_depth = agentIndex + 1, depth
        if agentIndex == gameState.getNumAgents() - 1:
            next_agentIndex, next_depth = 0, depth - 1
        min_value, min_action = float('inf'), Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            next_State = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxSearch(next_State, next_agentIndex, next_depth)[0]
            if value < min_value:
                min_value, min_action = value, action
        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState, 0, self.depth, -float('inf'), float('inf'))[1]
    
    def minimaxSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maximize(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.minimize(gameState, depth, agentIndex, alpha, beta)
        
    def maximize(self, gameState, depth, agentIndex, alpha, beta):
        next_agentIndex, next_depth = agentIndex + 1, depth
        if agentIndex == gameState.getNumAgents() - 1:
            next_agentIndex, next_depth = 0, depth - 1
        max_value, max_action = -float('inf'), Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            next_State = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxSearch(next_State, next_agentIndex, next_depth, alpha, beta)[0]
            if value > max_value:
                max_value, max_action = value, action
            if max_value > beta:
                return max_value, max_action
            alpha = max(alpha, max_value)
        return max_value, max_action
    
    def minimize(self, gameState, depth, agentIndex, alpha, beta):
        next_agentIndex, next_depth = agentIndex + 1, depth
        if agentIndex == gameState.getNumAgents() - 1:
            next_agentIndex, next_depth = 0, depth - 1
        min_value, min_action = float('inf'), Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            next_State = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxSearch(next_State, next_agentIndex, next_depth, alpha, beta)[0]
            if value < min_value:
                min_value, min_action = value, action
            if min_value < alpha:
                return min_value, min_action
            beta = min(beta, min_value)
        return min_value, min_action

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def Heuristic(self, gameState):
        if gameState.isWin():
            return 1
        elif gameState.isLose():
            return 0
        pacmanPos = gameState.getPacmanPosition()
        foodList = gameState.getFood().asList()
        minFoodDis = 0
        calcFoodNum = min(len(foodList), 10)
        for _ in range(calcFoodNum):
            index, minDis = 0, float('inf')
            for i in range(len(foodList)):
                dis = manhattanDistance(pacmanPos, foodList[i])
                if dis < minDis:
                    minDis, index = dis, i
            minFoodDis += minDis
            pacmanPos = foodList[index]
            foodList.pop(index)
        
        ghostStates = gameState.getGhostStates()
        minGhostDis = float('inf')
        for ghost in ghostStates:
            minGhostDis = min(manhattanDistance(pacmanPos, ghost.getPosition()), minGhostDis)
        if minGhostDis < 1:
            return 0
        
        # try a try, ac is best

        minFoodDisCoeff = -1.0
        minGhostDisCoeff = 1.0
        
        mysticValue = minFoodDisCoeff / (minFoodDis + 1) + minGhostDis * minGhostDisCoeff + gameState.getScore()

        return mysticValue >= 10
    
    def findCloseGhost(self, gameState):
        x0, y0 = gameState.getPacmanPosition()
        q = util.Queue()
        q.push((x0, y0, 0))
        vis = [(x0, y0)]
        ghostPos = gameState.getGhostPositions()
        wallPos = gameState.getWalls()

        while not q.isEmpty():
            x, y, dis = q.pop()
            for ghost in ghostPos:
                if manhattanDistance((x, y), ghost) < 1:
                    return dis
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if wallPos[nx][ny] or (nx, ny) in vis:
                    continue
                vis.append((nx, ny))
                q.push((nx, ny, dis + 1))
        return 10000
    
    def findCloseFood(self, gameState):
        x0, y0 = gameState.getPacmanPosition()
        vis = [(x0, y0)]
        foodPos = gameState.getFood().asList()
        capPos = gameState.getCapsules()
        wallPos = gameState.getWalls()

        q = util.Queue()
        for action in gameState.getLegalActions(0):
            if action == Directions.NORTH:
                nx, ny = x0, y0 + 1
            elif action == Directions.SOUTH:
                nx, ny = x0, y0 - 1
            elif action == Directions.EAST:
                nx, ny = x0 + 1, y0
            elif action == Directions.WEST:
                nx, ny = x0 - 1, y0
            else: 
                nx, ny = x0, y0
            if (nx, ny) in foodPos or (nx, ny) in capPos:
                return action
            q.push((nx, ny, action))
            vis.append((nx, ny))

        while not q.isEmpty():
            x, y, action = q.pop()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if wallPos[nx][ny] or (nx, ny) in vis:
                    continue
                if (nx, ny) in foodPos or (nx, ny) in capPos:
                    return action
                vis.append((nx, ny))
                q.push((nx, ny, action))

        return Directions.STOP

    def getAction(self, gameState):
        minGhostDis = 3
        if self.findCloseGhost(gameState) >= minGhostDis:
            return self.findCloseFood(gameState)
        self.totalAgents = gameState.getNumAgents()
        
        self.root = Node(gameState, None, None, 0)
        for _ in range(2000):
            # print("loop", _)
            nextNode = self.Selection(self.root)
            # print("loop", _)
            self.Expansion(nextNode)
            # print("loop", _)
            endNode, reward = self.Simulation(nextNode)
            # print("loop", _)
            self.Backpropagation(endNode, reward)

        bestChildren, maxRatio = [], -float('inf')
        for child in self.root.children:
            nowRatio = child.visitTimes / self.root.visitTimes
            if nowRatio > maxRatio:
                maxRatio, bestChildren = nowRatio, [child]
            elif nowRatio == maxRatio:
                bestChildren.append(child)
        if bestChildren == []:
            return random.choice(gameState.getLegalActions(0))
        return random.choice(bestChildren).action

    def Selection(self, nowNode):
        while nowNode.children != []:
            nowChildren = nowNode.children.copy()
            if len(nowChildren) != 1 or nowChildren[0].action != Directions.STOP:
                i = 0
                while i < len(nowChildren):
                    if nowChildren[i].action == Directions.STOP:
                        nowChildren.pop(i)
                    i += 1
           
            maxUCB, bestChildren = -float('inf'), []
            for child in nowChildren:
                if child.visitTimes == 0:
                    return child
                if nowNode.isPacman:
                    ucb = (child.totalReward / child.visitTimes) + sqrt(2.0 * log(nowNode.visitTimes) / child.visitTimes)
                else:
                    ucb = ((child.visitTimes - child.totalReward) / child.visitTimes) + sqrt(2.0 * log(nowNode.visitTimes) / child.visitTimes)
                if ucb > maxUCB:
                    maxUCB, bestChildren = ucb, [child]
                elif ucb == maxUCB:
                    bestChildren.append(child)       
            nowNode = random.choice(bestChildren)
        return nowNode
          
    def Expansion(self, fatherNode):
        nowAgentIndex = fatherNode.agentIndex
        actionsList = fatherNode.state.getLegalActions(nowAgentIndex)
        nextAgentIndex = (nowAgentIndex + 1) % self.totalAgents
        for action in actionsList:
            nextGameState = fatherNode.state.generateSuccessor(nowAgentIndex, action)
            childNode = Node(nextGameState, fatherNode, action, nextAgentIndex)
            fatherNode.children.append(childNode)

    def Simulation(self, nowNode):
        gameState, nowAgentIndex = nowNode.state, nowNode.agentIndex
        for _ in range(10):
            if gameState.isWin():
                return nowNode, 1
            elif gameState.isLose():
                return nowNode, 0
            else:
                validActions = gameState.getLegalActions(nowAgentIndex)
                if validActions == []:
                    return nowNode, 0
                action = random.choice(gameState.getLegalActions(nowAgentIndex))
                gameState = gameState.generateSuccessor(nowAgentIndex, action)
                nowAgentIndex = (nowAgentIndex + 1) % self.totalAgents
        # print(self.Heuristic(gameState))
        return nowNode, self.Heuristic(gameState)

    def Backpropagation(self, nowNode, nowReward):
        while nowNode is not None:
            nowNode.visitTimes += 1
            nowNode.totalReward += nowReward 
            nowNode = nowNode.parent       

class Node:
    def __init__(self, gameState, parent, action, agentIndex):
        self.state = gameState
        self.parent = parent
        self.action = action
        self.children = []
        self.totalReward = 0.0
        self.visitTimes = 0
        self.agentIndex = agentIndex
        self.isPacman = (agentIndex == 0)


    
            

