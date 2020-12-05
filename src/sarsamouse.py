import numpy as np
import random
import json

MOUSE_SAVE_FNAME = "mouse.json"

def getAvgOfSubMatrix(matrix, axis0, axis1):
    return np.average(matrix[np.ix_(axis0, axis1)])

class SarsaMouse:
    def __init__(self):
        self.alpha = 1
        self.gamma = 0.9
        self.epsilon = 1
        self.lam = 0.7
        self.eligibilityCutoff = 0.0001

        self.episodeCount = 0
        self.updateCount = 0
        self.k = 0
        self.lastK = 0

        self.lastStateActions = []
        self.actionCount = 8
        self.scentSpace = np.linspace(0.0, 255.0, 4).tolist()
        self.terrainSpace = np.linspace(-125.0, 125.0, 4).tolist()

        self.QE = {}

        try:
            with open(MOUSE_SAVE_FNAME) as json_file:
                data = json.load(json_file)
                self.load(data)
        except:
            pass

    def decayEpsilon(self, k):
        self.lastK = k
        self.epsilon = 1. / (self.k + k) ** .2
        self.alpha = 1. / (self.k + k) ** .2

    def update(self, reward):
        if reward != 0: print(f"R: {round(reward, 2)}")
        state, action = self.lastStateActions.pop()
        statePrime, actionPrime = self.lastStateActions[0]

        e = 1
        Q = 0
        QPrime = 0

        try:
            Q, e = self.QE[state, action]
            self.QE[state, action] = (Q, e + 1.0)
        except KeyError:
            Q, e = self.QE[state, action] = (0.0, 1.0)

        try:
            QPrime, ePrime = self.QE[statePrime, actionPrime]
        except KeyError:
            QPrime = 0

        err = reward + self.gamma * QPrime - Q

        for key in self.QE.keys():
            qValue, eligibilityTrace = self.QE[key]
            if eligibilityTrace == 0.0:
                continue
            newQ = qValue + self.alpha * err * eligibilityTrace
            newE = self.gamma * self.lam * eligibilityTrace
            self.QE[key] = (newQ, newE if newE > self.eligibilityCutoff else 0.0)

        self.updateCount += 1

    def getActionGreedily(self, state):
        # Get action based on best Q value
        actions = []
        #print("Available actions for state {}:".format(state))
        for action in range(self.actionCount):
            try:
                q = self.QE[state, action][0]
                actions.append(q)
            except KeyError:
                actions.append(0.0)
        if len(actions) == 0:
            return random.randint(0, self.actionCount - 1)
        choices = [i for i, x in enumerate(actions) if x == max(actions)]
        choice = random.choice(choices)
        #print("\nChoosing action {}\n\n".format(choice))
        return choice

    def getAction(self, sense):
        foodMatrix = sense.food_smell
        terrainSight = sense.elevation_sight
        currentTerrain = terrainSight[2][2]
        simplifiedTerrainSight = np.array([
            [currentTerrain - getAvgOfSubMatrix(terrainSight,[0,1],[0,1]),currentTerrain - getAvgOfSubMatrix(terrainSight,[0,1],[2]),currentTerrain - getAvgOfSubMatrix(terrainSight,[0,1],[3,4])],
            [currentTerrain - getAvgOfSubMatrix(terrainSight,[2],[0,1]),0.,currentTerrain - getAvgOfSubMatrix(terrainSight,[2],[3,4])],
            [currentTerrain - getAvgOfSubMatrix(terrainSight,[3,4],[0,1]),currentTerrain - getAvgOfSubMatrix(terrainSight,[3,4],[2]),currentTerrain - getAvgOfSubMatrix(terrainSight,[3,4],[3,4])],
        ])
        terrainState = self.getStateFromMatrix(simplifiedTerrainSight, self.terrainSpace)

        dangerSight = sense.danger_sight
        simplifiedDangerSight = np.array([
            [1. if 255. in dangerSight[np.ix_([0,2],[0,2])] else 0., 1. if 255. in dangerSight[np.ix_([0,2],[2,4])] else 0.],
            [1. if 255. in dangerSight[np.ix_([2,4],[0,2])] else 0., 1. if 255. in dangerSight[np.ix_([2,4],[2,4])] else 0.],
        ])
        dangerState = self.getStateFromMatrix(simplifiedDangerSight, [0.,1.], False)

        simplifiedFoodSmell = np.array([
            [getAvgOfSubMatrix(foodMatrix, [0,1], [0,1]),getAvgOfSubMatrix(foodMatrix, [0,1], [1,2])],
            [getAvgOfSubMatrix(foodMatrix, [1,2], [0,1]),getAvgOfSubMatrix(foodMatrix, [1,2], [1,2])]
        ])
        foodState = self.getStateFromMatrix(simplifiedFoodSmell, self.scentSpace)

        state = (foodState, dangerState, terrainState)

        # Decide whether we act greedily or explore randomly based on epsilon
        randValue = random.random()
        action = 0
        if randValue <= self.epsilon:
            action = random.randint(0, self.actionCount - 1)
        else:
            action = self.getActionGreedily(state)

        self.lastStateActions.append((state, action))
        return action

    def getStateFromMatrix(self, matrix, space, skipMiddle = True):
        sum = 0
        count = 0
        middleIndex = int(matrix.shape[0] / 2) * matrix.shape[1] + int(matrix.shape[1] / 2)
        for e in np.digitize(matrix, space).flat:
            if not (skipMiddle and count == middleIndex):  # ignore middle value because that's where our agent is
                sum += (e - 1) * len(space) ** count
                count += 1
        return int(sum)

    def save(self):
        # Saves to json file
        data = {
            "k": self.k + self.lastK,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda": self.lam,
            "epsilon": self.epsilon,
            "eligibilityCutoff": self.eligibilityCutoff,
            "updateCount": self.updateCount,
            "episodeCount": self.episodeCount + self.lastK,
            "possibleActions": self.actionCount,
            "lastStateActions": self.lastStateActions,
            "scentSpace": self.scentSpace,
            "terrainSpace": self.terrainSpace,
            "dataKeysState": [k[0] for k in self.QE.keys()],
            "dataKeysAction": [k[1] for k in self.QE.keys()],
            "dataValues": [v for v in self.QE.values()]
        }
        with open(MOUSE_SAVE_FNAME, 'w') as json_file:
            json.dump(data, json_file)

    def load(self, data):
        self.k = data["k"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.lam = data["lambda"]
        self.epsilon = data["epsilon"]
        self.eligibilityCutoff = data["eligibilityCutoff"]
        self.updateCount = data["updateCount"]
        self.episodeCount = data["episodeCount"]
        self.actionCount = data["possibleActions"]
        self.lastStateActions = [(tuple(s[0]), s[1]) for s in data["lastStateActions"]]
        self.scentSpace = data["scentSpace"]
        self.terrainSpace = data["terrainSpace"]
        qeKeysState = data["dataKeysState"]
        qeKeysAction = data["dataKeysAction"]
        qeValues = data["dataValues"]
        for i in range(len(qeValues)):
            keyAction = int(qeKeysAction[i])
            keyState = tuple(qeKeysState[i])
            value = tuple(qeValues[i])
            self.QE[keyState, keyAction] = value
