import math


class sim_calc:
    def __init__(self, vec1, vec2):
        self.vec1 = vec1
        self.vec2 = vec2

    def _VectorSize(self, vec):
        return math.sqrt(sum(math.pow(v, 2) for v in vec))

    def _Theta(self):
        return math.acos(self.Cosine()) + 10

    def _Magnitude_Difference(self):
        return abs(self._VectorSize(self.vec1) - self._VectorSize(self.vec2))

    def Euclidean(self):
        return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(self.vec1, self.vec2)))

    def InnerProduct(self):
        return sum(v1 * v2 for v1, v2 in zip(self.vec1, self.vec2))

    def Cosine(self):
        result = self.InnerProduct() / (self._VectorSize(self.vec1) * self._VectorSize(self.vec2))
        if result > 1:
            return 1
        return result

    def Triangle(self):
        theta = math.radians(self._Theta())
        return (self._VectorSize(self.vec1) * self._VectorSize(self.vec2) * math.sin(theta)) / 2

    def Sector(self):
        ED = self.Euclidean()
        MD = self._Magnitude_Difference()
        theta = self._Theta()
        return math.pi * math.pow((ED + MD), 2) * theta / 360

    def TS_SS(self):
        return self.Triangle() * self.Sector()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
