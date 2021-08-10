from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared = 50):
        self.nextObjID = 0
        self.objs = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objs[self.nextObjID] = centroid
        self.disappeared[self.nextObjID] = 0
        self.nextObjID += 1

    def deregister(self, objID):
        del self.objs[objID]
        del self.disappeared[objID]

    # i might just make another func to find obj closest to center

    def update(self, rects):

        if len(rects) == 0:
            for objID in list(self.disappeared.keys()):
                self.disappeared[objID] += 1

                if self.disappeared[objID] > self.maxDisappeared:
                    self.deregister(objID)

            return self.objs

        inputCentroids = np.zeros((len(rects), 2), dtype = 'int')

        for (i, (startx, starty, wid, hei)) in enumerate(rects):
            cx = int((startx + wid)/2.0)
            cy = int((starty + hei)/2.0)
            inputCentroids[i] = (cx, cy)

        if len(self.objs) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objIDs = list(self.objs.keys())
            objectCentroids = list(self.objs.values())

            d = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = d.min(axis = 1).argsort()

            cols = d.argmin(axis = 1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objID = objIDs[row]
                self.objs[objID] = inputCentroids[col]
                self.disappeared[objID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, d.shape[0])).difference(usedRows)
            unusedCols = set(range(0, d.shape[1])).difference(usedCols)

            if d.shape[0] >= d.shape[1]:
                for row in unusedRows:
                    objID = objIDs[row]
                    self.disappeared[objID] += 1

                    if self.disappeared[objID] > self.maxDisappeared:
                        self.deregister(objID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objs
    
