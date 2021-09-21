import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
import os
from pathlib import Path

plt.ioff()


class TourVisualizer:
    def __init__(self, base_dir, output_path, tours, time_slice, file_name, is_exp, **exp_params):

        self.source = 'r'
        self.base_dir = base_dir
        self.output_path = output_path
        self.map_berlin = gpd.read_file(os.path.join(
            self.base_dir, "input_data", "misc", "berlin_maps",'berlin.shp'))
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.figure_ax.set_facecolor('none')
        self.map_berlin.plot(ax=self.figure_ax, alpha=0.4, color='grey')
        self.df_all = pd.read_csv(os.path.join(
            self.base_dir, 'input_data', 'surve_mobility', 'orders', file_name + '_orders.csv'))
        self.df = self.df_all.copy()
        self.df.drop(self.df_all.index[0], inplace=True)
        self.df_depot = self.df_all.copy()
        self.df_depot.drop(self.df_all.index[1:], inplace=True)
        self.file_name = file_name
        if(is_exp):
            # self.folder_name_result = os.path.join(
            #     self.base_dir, "results", "vns", "surve_mobility", "experiments", file_name, exp_params["test_name"], "tour_visuals", str(exp_params["exp_id"]) if "exp_id" in exp_params else "")
            self.folder_name_result = os.path.join(
                self.output_path, "tour_visuals", str(exp_params["exp_id"]) if "exp_id" in exp_params else "")
            Path(self.folder_name_result).mkdir(parents=True, exist_ok=True)
        else:
            # self.folder_name_result = os.path.join(
            #     self.base_dir, "results", "vns", "surve_mobility", file_name, 'visualisation')
            self.folder_name_result = os.path.join(
                self.output_path, 'tour_visuals')
        self.time_slice = time_slice
        #self.path_queue = tours
        self.tours = tours
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._customer_color_end = 'darkgreen'
        self._line_color = 'seagreen'
        self._line_color_start_end = 'darksalmon'

    def _draw_point(self, scheduled_customers):
        geometry_start = [Point(xy) for xy in zip(
            self.df['XCOORD'], self.df['YCOORD'])]
        geometry_depot = [Point(xy) for xy in zip(
            self.df_depot['XCOORD'], self.df_depot['YCOORD'])]
        geometry_end = [Point(xy) for xy in zip(
            self.df['XCOORD_END'], self.df['YCOORD_END'])]
        geometry_all_start = [Point(xy) for xy in zip(
            self.df_all['XCOORD'], self.df_all['YCOORD'])]
        geometry_all_end = [Point(xy) for xy in zip(
            self.df_all['XCOORD_END'], self.df_all['YCOORD_END'])]

        self.geo_df_depot = gpd.GeoDataFrame(
            self.df_depot, crs='epsg:4326', geometry=geometry_depot).copy()
        self.geo_df_start = gpd.GeoDataFrame(
            self.df, crs='epsg:4326', geometry=geometry_start).copy()
        self.geo_df_end = gpd.GeoDataFrame(
            self.df, crs='epsg:4326', geometry=geometry_end).copy()

        self.geo_df_all_start = gpd.GeoDataFrame(
            self.df_all, crs='epsg:4326', geometry=geometry_all_start).copy()

        for x, y, label in zip(self.geo_df_all_start.iloc[scheduled_customers].geometry.x, self.geo_df_all_start.iloc[scheduled_customers].geometry.y, self.geo_df_all_start.iloc[scheduled_customers].geometry.index):
            self.figure_ax.annotate(str(label), xy=(x, y), xytext=(
                1, 1), textcoords='offset points', fontsize=6)

        self.geo_df_all_start.drop(
            self.geo_df_all_start.index[0], inplace=True)
        self.geo_df_all_start.reset_index(inplace=True)

        self.geo_df_all_end = gpd.GeoDataFrame(
            self.df_all, crs='epsg:4326', geometry=geometry_all_end).copy()
        self.geo_df_all_end.drop(self.geo_df_all_end.index[0], inplace=True)
        self.geo_df_all_end.reset_index(inplace=True)
        scheduled_customers_index = [i - 1 for i in scheduled_customers]
        self.geo_df_depot.plot(
            ax=self.figure_ax, markersize=5, color='red', marker='o', label='Depot')
        self.geo_df_end.iloc[scheduled_customers_index].plot(ax=self.figure_ax, markersize=5,
                                                             color='green', marker='o', label='End')
        self.geo_df_start.iloc[scheduled_customers_index].plot(
            ax=self.figure_ax, markersize=5, color='blue', marker='o', label='Start')

        self.geo_df_all_start = gpd.GeoDataFrame(
            self.df_all, crs='epsg:4326', geometry=geometry_all_start).copy()
        self.geo_df_all_end = gpd.GeoDataFrame(
            self.df_all, crs='epsg:4326', geometry=geometry_all_end).copy()

        plt.pause(0.5)

    @staticmethod
    def _get_scheduled_customers(tours):
        scheduled_customers = []
        for tour in tours:
            for customer in tour:
                if not customer in scheduled_customers and customer != 0:
                    scheduled_customers.append(customer)
        return scheduled_customers

    def run(self):
        # 先绘制出各个结点 Plot the nodes first.
        scheduled_customers = self._get_scheduled_customers(self.tours)
        self._draw_point(scheduled_customers)
        for tour in self.tours:
            self._draw_line(tour)

        current_time = time.strftime('%H:%M:%S', time.localtime())
        self.figure_ax.set_title(
            'Tour visualization at time slice ' + str(self.time_slice))
        file_name_result = 'result_visualization.png'
        plt.savefig(os.path.join(self.folder_name_result, file_name_result.split(
            '.')[0] + '_' + str(self.time_slice) + '.png'), dpi=400)
        plt.close()
        # self.figure.show()

        # 从队列中读取新的path，进行绘制 Read the new path from the queue and draw it.
        '''while True:
            if not self.path_queue.empty():
                # 取队列中最新的一个path，其他的path丢弃 takes the newest path in the queue and discards the other paths.
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

                # 重新绘制line Redrawing the line
                current_time = time.strftime('%H:%M:%S', time.localtime())
                self.figure_ax.set_title('travel distance: %0.2f, number of vehicles: %d , time: %s' % (distance, used_vehicle_num, current_time))
                self._draw_line(path)
                self.figure.show()
                result_folder_vis = self.folder_name_result + '/visualization'
                file_name_result = 'result_visualization.png'
                plt.savefig(os.path.join(result_folder_vis, file_name_result.split('.')[0] + '_' + str(self.time_slice) + '.png'))

            plt.pause(1)'''

    def _draw_line(self, path):
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
            self.figure_ax.plot(
                x_list, y_list, color=self._line_color, linewidth=0.5, label='line')

            x_list_start_end = [x_value_3, x_value_4]
            y_list_start_end = [y_value_3, y_value_4]
            self.figure_ax.plot(x_list_start_end, y_list_start_end, color=self._line_color_start_end, linewidth=0.5,
                                label='line')
