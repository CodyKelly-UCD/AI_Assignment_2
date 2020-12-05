from unittest import TestCase
from src import sarsamouse


class Test(TestCase):
    def test_create_spaces(self):
        testSpace = []
        for i in range(2):
            for j in range(2):
                testSpace.append((i, j))
        print(sarsamouse.createSpaces([[0, 0, 0], [1, 1], [2, 2]]))
        #print(testSpace)
