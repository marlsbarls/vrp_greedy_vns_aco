from string import Template
import os
import pandas as pd
import vns.src.config.preprocessing_config as prep_cfg
import datetime
from pathlib import Path
import copy


class TourPlanVisualizer:
    def __init__(self, base_dir, file, travel_time_matrix, planning_df, tours, cost, idle_time, is_exp, **exp_params):
        self.tours = tours
        self.cost = cost
        self.idle_time = idle_time
        self.travel_time_matrix = travel_time_matrix
        self.planning_df = planning_df
        self.base_dir = base_dir
        self.html_file = self._get_html_template()
        self.html_template = Template(self.html_file)
        if (is_exp):
            self.output_folder = os.path.join(
                self.base_dir, "experiments", "results", file, exp_params["test_name"], "tour_plans", str(exp_params["exp_id"]) if "exp_id" in exp_params else "")
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            self.output_file = os.path.join(self.output_folder, "schedule_" +
                                            str(exp_params["time"]) + ".html"
                                            )
        else:
            self.output_file = os.path.join(
                self.base_dir, "data", "results_optimization", file, "tour_plans", "schedule_" + file + ".html")

    def _get_html_template(self):
        html_file = open(os.path.join(self.base_dir, "data",
                                      "misc", "schedule_template.html"), "r")
        content = html_file.read()
        html_file.close()
        return content

    def _build_html(self, data, cost, number_of_tours, idle_time):
        outcome = self.html_template.safe_substitute(
            content=data, minimal_cost=str(round(cost, 2)), number_of_tours=number_of_tours, idle_time=round(idle_time, 2))
        output_file = open(self.output_file, "w+")
        output_file.write(outcome)
        output_file.close()

    def _build_rows(self):
        planning_df = self.planning_df[self.planning_df["SCHEDULED_TOUR"].isnull(
        ) != True]
        planning_df["SCHEDULED_TOUR"] = planning_df["SCHEDULED_TOUR"].astype(
            "int")
        planning_df = planning_df.sort_values(
            ["SCHEDULED_TOUR", "SCHEDULED_TIME"])
        u = copy.deepcopy(planning_df["SCHEDULED_TOUR"]).unique()

        rowTemplate = Template(
            "['$tourId', '$pathName', '$color', new Date(0, 0, 0, $start_hour, $start_minute, 0), new Date(0, 0, 0, $end_hour, $end_minute, 0)],")

        content = ""
        base_date = datetime.datetime(2000, 1, 1, 9)
        for tour in sorted(u):
            df = copy.deepcopy(
                planning_df[planning_df["SCHEDULED_TOUR"] == tour])
            # df.sort_values(["SCHEDULED_TIME"], inplace=True)
            prev_idx = -1
            prev_end_date = None
            for idx, order in df.iterrows():
                # Actual Order (Servicetime)
                start_date = base_date + \
                    datetime.timedelta(minutes=order["SCHEDULED_TIME"])
                end_date = start_date + \
                    datetime.timedelta(minutes=order["SERVICETIME"])
                content += rowTemplate.safe_substitute(tourId="Tour " + str(tour), pathName=order["CUST_NO"], color="7d7", start_hour=start_date.strftime(
                    "%H"), start_minute=start_date.strftime("%M"), end_hour=end_date.strftime("%H"), end_minute=end_date.strftime("%M"))

                # Previous Travel to Order
                if(prev_idx != -1):
                    # Travel time
                    travel_start_date = start_date - datetime.timedelta(
                        minutes=self.travel_time_matrix[df.loc[prev_idx]["order_id"] + ":" + order["order_id"]])
                    # Idle time
                    idle_end_date = travel_start_date
                    idle_start_date = prev_end_date
                    if(idle_end_date - idle_start_date >= datetime.timedelta(minutes=1)):
                        content += rowTemplate.safe_substitute(tourId="Tour " + str(tour), pathName="Idle", color="#e6786a", start_hour=idle_start_date.strftime(
                            "%H"), start_minute=idle_start_date.strftime("%M"), end_hour=idle_end_date.strftime("%H"), end_minute=idle_end_date.strftime("%M"))
                    # travel_start_date = start_date - datetime.timedelta(minutes=order["SCHEDULED_TIME"] - (
                    #     df.loc[prev_idx]["SCHEDULED_TIME"] + df.loc[prev_idx]["SERVICETIME"]))
                # Depot to first order
                else:
                    travel_start_date = start_date - \
                        datetime.timedelta(
                            minutes=self.travel_time_matrix["order_0:" + order["order_id"]])
                travel_end_date = start_date
                # Draw line
                if(travel_end_date - travel_start_date >= datetime.timedelta(minutes=1)):
                    content += rowTemplate.safe_substitute(tourId="Tour " + str(tour), pathName="Travel", color="d3d3d3", start_hour=travel_start_date.strftime(
                        "%H"), start_minute=travel_start_date.strftime("%M"), end_hour=travel_end_date.strftime("%H"), end_minute=travel_end_date.strftime("%M"))
                prev_idx = idx
                prev_end_date = end_date

            # Last order to depot
            drive_home_start_date = end_date
            drive_home_end_date = drive_home_start_date + \
                datetime.timedelta(
                    minutes=self.travel_time_matrix[order['order_id'] + ":" + "order_0"])
            content += rowTemplate.safe_substitute(tourId="Tour " + str(tour), pathName="Travel", color="d3d3d3", start_hour=drive_home_start_date.strftime(
                "%H"), start_minute=drive_home_start_date.strftime("%M"), end_hour=drive_home_end_date.strftime("%H"), end_minute=drive_home_end_date.strftime("%M"))

        return content

    def create_tour_plan(self):
        data = self._build_rows()
        cost = self.cost
        number_of_tours = len(self.tours)
        idle_time = self.idle_time
        self._build_html(data, cost, number_of_tours, idle_time)
