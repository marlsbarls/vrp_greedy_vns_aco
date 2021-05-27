'''
Helper class to quickly find nearest POIs of one kind.
Code inspired by https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
'''

from sklearn.neighbors import BallTree
import numpy as np
from shapely.geometry import Point, MultiPoint


class RangeFinder:
    def __init__(self, destiny_gdf):
        self.destiny_geom_col = destiny_gdf.geometry.name
        self.destiny = destiny_gdf.copy().reset_index(drop=True)
        self.destiny_radians = np.array(self.destiny[self.destiny_geom_col].apply(
            lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
        self.tree = BallTree(self.destiny_radians,
                             leaf_size=15, metric='haversine')

    def get_nearest(self, src_points, k_neighbours=1):
        distances, indices = self.tree.query(src_points, k=k_neighbours)

        # Transpose to get distances and indices into arrays
        distances = distances.transpose()
        indices = indices.transpose()

        # Get closest indices and distances (i.e. array at index 0)
        # note: for the second closest points, you would take index 1, etc.
        closest = indices[0:k_neighbours]
        # print(closest)
        closest_dist = distances[0:k_neighbours]

        # Return indices and distances
        return (closest, closest_dist)

    def nearest_neighbor(self, left_gdf, geo_col=None, return_dist=False, n_nearest=1):
        """
        For each point in left_gdf, find closest point in right GeoDataFrame and return them.

        NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
        """

        left_geom_col = left_gdf.geometry.name

        # Parse coordinates from points and insert them into a numpy array as RADIANS
        left_radians = np.array(left_gdf[left_geom_col].apply(
            lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

        # Find the nearest points
        # -----------------------
        # closest ==> index in right_gdf that corresponds to the closest point
        # dist ==> distance between the nearest neighbors (in meters)

        closest, dist = self.get_nearest(
            src_points=left_radians, k_neighbours=n_nearest)
        # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame

        closest_points = []
        for a in closest:
            closest_points.append(self.destiny.loc[a].reset_index(drop=True))
            # Add distance if requested
            if return_dist:
                # Convert to meters from radians
                earth_radius = 6371000  # meters
                closest_points[a]['distance'] = dist * earth_radius

        return closest_points
