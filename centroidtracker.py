from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.originRects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect):
        self.originRects[self.nextObjectID] = rect
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.originRects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_id(self, rect):
        (x, y, eX, eY) = rect
        cX = ((x + eX) / 2.0)
        cY = ((y + eY) / 2.0)

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        
        D = dist.cdist(np.array(objectCentroids), [(cX, cY)])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        objectID = None

        for (row, col) in zip(rows, cols):
            objectID = objectIDs[row]
            break
        return objectID

    def update(self, rects):

        if(len(rects) == 0):
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if(self.disappeared[objectID] > self.maxDisappeared):
                    self.deregister(objectID)
            return self.objects, self.originRects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for(i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if(len(self.objects) == 0):
            for i in range(0, len(inputCentroids)):
                centroid = inputCentroids[i]
                rect = rects[i]
                self.register(centroid, rect)
        
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):

                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.originRects[objectID] = rects[col]
                self.disappeared[objectID] = 0

                #add the row  and col  to their respective usedRows  and usedCols  sets
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:

                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                        
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object

            else:

                for col in unusedCols:
                    centroid = inputCentroids[col]
                    rect = rects[col]
                    self.register(centroid, rect)

        return self.objects, self.originRects
