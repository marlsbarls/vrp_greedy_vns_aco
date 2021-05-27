import os
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Load data.
path = './additional_data/graphs_statistics/date_order_count_grouped.csv'
df = pd.read_csv(path,
             sep=',',
             header=None,
             names=['date',
                    'orders'])

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

dates = []
orders = []
orders_2019 = []
orders_2020 = []

for i in range(len(df)):
    dates.append(df.iloc[i,0])
    orders.append(df.iloc[i,1])
    if df.iloc[i,0].year == 2019:
        orders_2019.append(df.iloc[i,1])
    else:
        orders_2020.append(df.iloc[i,1])


def line_graph():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot_date(dates, orders, linestyle='solid', markersize=2)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Orders per day',fontsize=20)
    fig.autofmt_xdate()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.savefig('./additional_data/graphs_statistics/orders_line_diagram.png')
    plt.show()

def boxplot():
    data_to_plot = [orders, orders_2019, orders_2020]
    fig = plt.figure(1, figsize=(20, 12))
    ax = fig.add_subplot(111)
    ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Overall', '2019', '2020'])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    plt.ylabel('Orders per day', fontsize=20)
    plt.savefig('./additional_data/graphs_statistics/orders_boxplot_comparison.png')
    plt.show()



def std():
    std_all = np.std(orders)
    std_2019 = np.std(orders_2019)
    std_2020 = np.std(orders_2020)

    print("Standard Deviation Orders Overall:", std_all, "\nStandard Deviation Orders 2019:", std_2019, "\nStandard Deviation Orders 2020:", std_2020)



line_graph()
boxplot()
std()
