import csv
import numpy as np
import random
from matplotlib import pyplot as plt

FILE_PATH = '/home/mroglan/gt/CS 4641/project/data/Column_Cleaned_Data.csv' # Your file path
EPS = 0.03
MIN_PTS = 1000
PROPORTION = 0.3
COLORS = ['r', 'g', 'b', 'm', 'y', 'k', 'c']


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """

        x_sum = np.sum(
            np.square(x),
            axis=1
        )[:,np.newaxis]

        y_sum = np.sum(
            np.square(y),
            axis=1
        )[np.newaxis,:]

        xy2 = np.squeeze((2 * np.dot(x, y[:,:,np.newaxis])), axis=2)

        total_sum = x_sum + y_sum - xy2

        return np.sqrt(total_sum)

class LocationDBScan:

    def __init__(self):

        self.locations = list()

        with open(FILE_PATH) as file:
            reader = csv.DictReader(file)

            for row in reader:
                lat = float(row['Latitude'])
                long = float(row['Longitude'])

                if random.random() < PROPORTION:
                    self.locations.append([lat, long])
        
        self.locations = np.array(self.locations)
    
    def plot_raw(self):
        plt.scatter(self.locations[:,0], self.locations[:,1])        
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.show()
    
    def plot_clustered(self):
        plots = list()
        for p in range(self.num_clusters + 2):
            plots.append([[], []])
        
        for i in range(self.locations.shape[0]):
            if self.cluster_idx[i] == -1:
                c = len(plots) - 1
            else:
                c = self.cluster_idx[i]
            plots[c][0].append(self.locations[i,0])
            plots[c][1].append(self.locations[i,1])
        
        for i, plot in enumerate(plots):
            if i == len(plots) - 1:
                plt.scatter(plot[0], plot[1], c='darkgrey')
            else:
                plt.scatter(plot[0], plot[1], c=COLORS[i % len(COLORS)])
        
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.show()
    
    def run(self):
        print(self.locations.shape)

        cluster_idx = np.full((self.locations.shape[0],), -1)
        visited = set()
        C = -1
        self.count = 0
        for p in cluster_idx:
            self.count += 1
            print(f'{self.count} / {self.locations.shape[0]}')
            if p in visited:
                continue
            visited.add(p)
            neighbors = self.regionQuery(p)
            # print('finished region query')
            if neighbors.shape[0] >= MIN_PTS:
                C += 1
                self.expandCluster(p, neighbors, C, cluster_idx, visited)   
        
        print(f'C: {C}')
        self.num_clusters = C + 1

        self.cluster_idx = cluster_idx
    
    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        cluster_idx[index] = C
        i = 0
        while i < len(neighborIndices):
            p = neighborIndices[i]
            i += 1
            if not p in visitedIndices:
                # print(C)
                visitedIndices.add(p)
                neighbor_pts = self.regionQuery(p)
                if neighbor_pts.shape[0] >= MIN_PTS:
                    self.count += 1
                    print(f'{self.count} / {self.locations.shape[0]}')
                    neighborIndices =  np.concatenate((neighborIndices, neighbor_pts))
            if cluster_idx[p] == -1:
                cluster_idx[p] = C

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """

        dists = pairwise_dist(self.locations[pointIndex][np.newaxis,:], self.locations)

        indices = np.argwhere(dists[0] <= EPS)

        indices = np.squeeze(indices, axis=1)

        return indices 



if __name__ == '__main__':
    scan = LocationDBScan()
    # scan.plot_raw()
    scan.run()
    scan.plot_clustered()