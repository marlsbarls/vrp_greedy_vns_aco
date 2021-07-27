import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
import os


class VrptwAcoFigure:
    def __init__(self, source, time_slice, nodes: list, path_queue: MPQueue, path_map, file_path, folder_name_result):
        """
        matplotlib绘图计算需要放在主线程，寻找路径的工作建议另外开一个线程，
        The matplotlib drawing calculations need to be put in the main thread, and the work of finding paths suggests a separate thread.
        当寻找路径的线程找到一个新的path的时候，将path放在path_queue中，图形绘制线程就会自动进行绘制
        When the thread looking for a path finds a new path, place the path in path_queue and the graphing thread will automatically draw it.
        queue中存放的path以PathMessage（class）的形式存在
        The path stored in the queue is in the form of PathMessage (class)
        nodes中存放的结点以Node（class）的形式存在，主要使用到Node.x, Node.y 来获取到结点的坐标
        The nodes in nodes are in the form of Node(class), which mainly uses Node.x, Node.y to get the coordinates of the node

        :param nodes: nodes是各个结点的list，包括depot
        :param path_queue: queue用来存放工作线程计算得到的path，队列中的每一个元素都是一个path，path中存放的是各个结点的id
        """

        self.nodes = nodes
        self.source = source

        # Distinction between test instances added
        if source == 't':
            self.figure = plt.figure(figsize=(10, 10))
            self.figure_ax = self.figure.add_subplot(1, 1, 1)
        elif source == 'r':
            self.map_berlin = gpd.read_file(path_map)
            self.figure = plt.figure(figsize=(10, 10))
            self.figure_ax = self.figure.add_subplot(1, 1, 1)
            self.map_berlin.plot(ax=self.figure_ax, alpha=0.4, color='grey')
            # l 39-48 modified
            self.df_all = pd.read_csv(file_path, usecols=['CUST_NO', 'XCOORD', 'YCOORD',   
                                                      'DEMAND', 'READYTIME', 'DUETIME', 'SERVICETIME',
                                                      'XCOORD_END', 'YCOORD_END'])
            self.df_all = self.df_all.rename(columns={'CUST_NO': 'CUST_NO.', 'XCOORD': 'YCOORD.', 'YCOORD': 'XCOORD.', 
                                              'READYTIME': 'READY_TIME', 'DUETIME': 'DUE_DATE', 
                                              'SERVICETIME': 'SERVICE_TIME', 'XCOORD_END': 'YCOORD_END', 'YCOORD_END': 'XCOORD_END'})
            self.df_all = self.df_all[['CUST_NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY_TIME', 'DUE_DATE',
                               'SERVICE_TIME', 'XCOORD_END', 'YCOORD_END']]
            self.df = self.df_all.drop(self.df_all.index[0])
            self.df_depot = self.df_all.drop(self.df_all.index[1:])

        
            

        self.folder_name_result = folder_name_result
        self.time_slice = time_slice
        self.path_queue = path_queue
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._customer_color_end = 'darkgreen'
        self._line_color = 'seagreen'
        self._line_color_start_end = 'darksalmon'

    def _draw_point(self):
        # 画出
        # depot
        if self.source == 't':
            self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], c=self._depot_color, label='depot', s=40)

            # 画出
            # customer
            # MOD: Add points for end location
            self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                                   list(node.y for node in self.nodes[1:]), c=self._customer_color, label='customer_start', s=20)

            self.figure_ax.scatter(list(node.x_end for node in self.nodes[1:]),
                                   list(node.y_end for node in self.nodes[1:]), c=self._customer_color_end, label='customer_end', s=20)

            for i, label in enumerate(list(node.id for node in self.nodes[0:])):
                self.figure_ax.annotate(str(label)+'s', (self.nodes[i].x, self.nodes[i].y), fontsize=6)

            for i, label in enumerate(list(node.id for node in self.nodes[0:])):
                if self.nodes[i].x != self.nodes[i].x_end:
                    self.figure_ax.annotate(str(label)+'e', (self.nodes[i].x_end, self.nodes[i].y_end), fontsize=6)

            plt.pause(0.01)

        # MOD: Add procedure for Chargery Instances
        elif self.source == 'r':
            geometry_start = [Point(xy) for xy in zip(self.df['YCOORD.'], self.df['XCOORD.'])]
            geometry_depot = [Point(xy) for xy in zip(self.df_depot['YCOORD.'], self.df_depot['XCOORD.'])]
            geometry_end = [Point(xy) for xy in zip(self.df['YCOORD_END'], self.df['XCOORD_END'])]
            geometry_all_start = [Point(xy) for xy in zip(self.df_all['YCOORD.'], self.df_all['XCOORD.'])]
            geometry_all_end = [Point(xy) for xy in zip(self.df_all['YCOORD_END'], self.df_all['XCOORD_END'])]

            self.geo_df_depot = gpd.GeoDataFrame(self.df_depot, crs='epsg:4326', geometry=geometry_depot).copy()
            self.geo_df_start = gpd.GeoDataFrame(self.df, crs='epsg:4326', geometry=geometry_start).copy()
            self.geo_df_end = gpd.GeoDataFrame(self.df, crs='epsg:4326', geometry=geometry_end).copy()

            self.geo_df_all_start = gpd.GeoDataFrame(self.df_all, crs='epsg:4326', geometry=geometry_all_start).copy()

            for x, y, label in zip(self.geo_df_all_start.geometry.x, self.geo_df_all_start.geometry.y, self.geo_df_all_start.geometry.index):
                self.figure_ax.annotate(str(label), xy=(x, y), xytext=(1, 1), textcoords='offset points', fontsize=6)

            self.geo_df_all_start.drop(self.geo_df_all_start.index[0], inplace=True)
            self.geo_df_all_start.reset_index(inplace=True)

            self.geo_df_all_end = gpd.GeoDataFrame(self.df_all, crs='epsg:4326', geometry=geometry_all_end).copy()
            self.geo_df_all_end.drop(self.geo_df_all_end.index[0], inplace=True)
            self.geo_df_all_end.reset_index(inplace=True)

            self.geo_df_depot.plot(ax=self.figure_ax, markersize=5, color='red', marker='o', label='Depot')
            self.geo_df_start.plot(ax=self.figure_ax, markersize=5, color='blue', marker='o', label='Start')
            self.geo_df_end.plot(ax=self.figure_ax, markersize=5, color='green', marker='o', label='End')

            self.geo_df_all_start = gpd.GeoDataFrame(self.df_all, crs='epsg:4326', geometry=geometry_all_start).copy()
            self.geo_df_all_end = gpd.GeoDataFrame(self.df_all, crs='epsg:4326', geometry=geometry_all_end).copy()

            plt.pause(0.01)

    def run(self):
        # 先绘制出各个结点
        # Plot the nodes first.
        self._draw_point()
        #self.figure.show()

        # 从队列中读取新的path，进行绘制
        # Read the new path from the queue and draw it.
        while True:
            if not self.path_queue.empty():
                # 取队列中最新的一个path，其他的path丢弃
                # takes the newest path in the queue and discards the other paths.
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    plt.close(self.figure)
                    break

                # 需要先记录要移除的line，不能直接在第一个循环中进行remove，
                # Needs to record the line to be removed first, can't just remove it in the first loop.
                # 不然self.figure_ax.lines会在循环的过程中改变，导致部分line无法成功remove
                # Otherwise, self.figure_ax.lines will change during the loop, resulting in a partial line not being successfully removed!
                remove_obj = []
                for line in self.figure_ax.lines:
                    if line._label == 'line':
                        remove_obj.append(line)

                for line in remove_obj:
                    self.figure_ax.lines.remove(line)
                remove_obj.clear()

                # 重新绘制line
                # Redrawing the line
                # MOD: Save figure to visualization folder in result folder
                current_time = time.strftime('%H:%M:%S', time.localtime())
                self.figure_ax.set_title('travel distance: %0.2f, number of vehicles: %d , time: %s' % (distance, used_vehicle_num, current_time))
                self._draw_line(path)
                # self.figure.show()
                result_folder_vis = self.folder_name_result + '/visualization'
                file_name_result = 'result_visualization.png'
                plt.savefig(os.path.join(result_folder_vis, file_name_result.split('.')[0] + '_' + str(self.time_slice) + '.png'))

            plt.pause(1)

    def _draw_line(self, path):
        if self.source == 't':
            # 根据path中index进行路径的绘制
            # Draw the path from the index in the path
            for i in range(1, len(path)):
                x_list = [self.nodes[path[i - 1]].x_end, self.nodes[path[i]].x]
                y_list = [self.nodes[path[i - 1]].y_end, self.nodes[path[i]].y]
                self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=1.5, label='line')

                x_list_start_end = [self.nodes[path[i]].x, self.nodes[path[i]].x_end]
                y_list_start_end = [self.nodes[path[i]].y, self.nodes[path[i]].y_end]
                self.figure_ax.plot(x_list_start_end, y_list_start_end, color=self._line_color_start_end, linewidth=0.5, label='line')

        # MOD: Add procedure for Chargery Instances
        elif self.source == 'r':
            x_value_1 = None
            x_value_2 = None
            x_value_3 = None
            x_value_4 = None
            y_value_1 = None
            y_value_2 = None
            y_value_3 = None
            y_value_4 = None

            for i in range(1, len(path)):
                if path[i] == 0:
                    x_value_1 = self.geo_df_all_end.geometry.x[path[i-1]]
                    y_value_1 = self.geo_df_all_end.geometry.y[path[i-1]]
                    x_value_2 = self.geo_df_depot.geometry.x[0]
                    y_value_2 = self.geo_df_depot.geometry.y[0]
                    x_value_3 = x_value_2
                    y_value_3 = y_value_2
                    x_value_4 = x_value_2
                    y_value_4 = y_value_2

                elif path[i-1] == 0:
                    x_value_1 = self.geo_df_depot.geometry.x[0]
                    y_value_1 = self.geo_df_depot.geometry.y[0]
                    x_value_2 = self.geo_df_all_start.geometry.x[path[i]]
                    y_value_2 = self.geo_df_all_start.geometry.y[path[i]]
                    x_value_3 = x_value_2
                    y_value_3 = y_value_2
                    x_value_4 = self.geo_df_all_end.geometry.x[path[i]]
                    y_value_4 = self.geo_df_all_end.geometry.y[path[i]]

                elif path[i-1] != 0 and path[i] != 0:
                    x_value_1 = self.geo_df_all_end.geometry.x[path[i - 1]]
                    y_value_1 = self.geo_df_all_end.geometry.y[path[i - 1]]
                    x_value_2 = self.geo_df_all_start.geometry.x[path[i]]
                    y_value_2 = self.geo_df_all_start.geometry.y[path[i]]
                    x_value_3 = x_value_2
                    y_value_3 = y_value_2
                    x_value_4 = self.geo_df_all_end.geometry.x[path[i]]
                    y_value_4 = self.geo_df_all_end.geometry.y[path[i]]

                x_list = [x_value_1, x_value_2]
                y_list = [y_value_1, y_value_2]
                self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=0.5, label='line')

                x_list_start_end = [x_value_3, x_value_4]
                y_list_start_end = [y_value_3, y_value_4]
                self.figure_ax.plot(x_list_start_end, y_list_start_end, color=self._line_color_start_end, linewidth=0.5,
                                    label='line')
