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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score =  successorGameState.getScore()

        # Calculate the distance to the closest food
        minFoodDistance = float('inf')
        for food in newFoodList:
            minFoodDistance = min(minFoodDistance, util.manhattanDistance(newPos, food))
        score += 1.0 / minFoodDistance

        # Calculate the effect of ghost proximity
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distanceToGhost = util.manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                score += 10.0 / distanceToGhost
            else:
                # If the ghost is not scared, approaching is bad
                if distanceToGhost <= 1:
                    score -= 20  # Large penalty for being too close to a live ghost

        # Discourage stopping
        if action == Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (question 2)
    """
    def maxValue(self, gameState: GameState, agentIndex: int, depth: int):
        score = float('-inf')
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score = max(score, self.minimax(successorGameState, agentIndex + 1, depth))
        
        return score
    
    def minValue(self, gameState: GameState, agentIndex: int, depth: int):
        score = float('inf')
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score = min(score, self.minimax(successorGameState, agentIndex + 1, depth))
        
        return score

    def minimax(self, gameState: GameState, agentIndex: int, depth: int):
        # Get the number of agents
        numAgents = gameState.getNumAgents()
        agentIndex = agentIndex % numAgents
        if agentIndex == 0:
            depth += 1

        # Check if the game state is a terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Get the legal actions for the pacman
        legalActions = gameState.getLegalActions(0)
        # Initialize the best score and best action
        bestScore = float('-inf')
        bestAction = None
        # Loop through all the legal actions
        for action in legalActions:
            # Get the successor game state
            successorGameState = gameState.generateSuccessor(0, action)
            # Get the score for the successor game state
            score = self.minimax(successorGameState, 1, 1)
            # Update the best score and best action
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        score = float('inf')
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score = min(score, self.minimax(successorGameState, agentIndex + 1, depth, alpha, beta))
            if score < alpha:
                return score
            beta = min(beta, score)
        
        return score

    def maxValue(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        score = float('-inf')
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score = max(score, self.minimax(successorGameState, agentIndex + 1, depth, alpha, beta))
            if score > beta:
                return score
            alpha = max(alpha, score)

        return score

    def minimax(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        # Get the number of agents
        numAgents = gameState.getNumAgents()
        agentIndex = agentIndex % numAgents
        if agentIndex == 0:
            depth += 1

        # Check if the game state is a terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Get the legal actions for the pacman
        legalActions = gameState.getLegalActions(0)
        # Initialize the best score and best action
        bestScore = float('-inf')
        bestAction = None
        # Loop through all the legal actions
        for action in legalActions:
            # Get the successor game state
            successorGameState = gameState.generateSuccessor(0, action)
            # Get the score for the successor game state
            score = self.minimax(successorGameState, 1, 1, bestScore, float('inf'))
            # Update the best score and best action
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expValue(self, gameState: GameState, agentIndex: int, depth: int):
        score = 0
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score += self.expectimax(successorGameState, agentIndex + 1, depth)
        
        return score / len(legalActions)

    def maxValue(self, gameState: GameState, agentIndex: int, depth: int):
        score = float('-inf')
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            score = max(score, self.expectimax(successorGameState, agentIndex + 1, depth))
        
        return score
        

    def expectimax(self, gameState: GameState, agentIndex: int, depth: int):
        numAgents = gameState.getNumAgents()
        agentIndex = agentIndex % numAgents
        if agentIndex == 0:
            depth += 1
        
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Get the legal actions for the pacman
        legalActions = gameState.getLegalActions(0)
        # Initialize the best score and best action
        bestScore = float('-inf')
        bestAction = None
        # Loop through all the legal actions
        for action in legalActions:
            # Get the successor game state
            successorGameState = gameState.generateSuccessor(0, action)
            # Get the score for the successor game state
            score = self.expectimax(successorGameState, 1, 1)
            # Update the best score and best action
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
