import pandas as pd
from collections import Counter
import numpy as np

df_vns = pd.read_csv('/Users/marlenebuhl/Documents/University/thesis/coding_part/thesis_marlene/input_data/surve_mobility/orders/original/2020-07-07_orders.csv')
df_aco = pd.read_csv('/Users/marlenebuhl/Downloads/VRPTW-ACO-python/test_files/test-07-07.csv')
order_ids_aco = df_aco['order_id'].to_list()
dict_order_ids = Counter(order_ids_aco).keys()
order_ids_aco = list(dict_order_ids)
cust_no = 0
for index, row in df_vns.iterrows():
    order_id_row = row['order_id'].split('_')[1]
    if order_id_row not in order_ids_aco:
        df_vns.drop(index, inplace=True )
    else:
        df_vns.at[index, 'CUST_NO'] = cust_no
        cust_no += 1

# df_vns['XCOORD'] = df_aco['XCOORD.']
# df_vns['YCOORD'] = df_aco['YCOORD.']
# df_vns['YCOORD_END'] = df_aco['YCOORD_END']
# df_vns['XCOORD_END'] = df_aco['XCOORD_END']


df_vns.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57], axis='index', inplace=True)
# order_ids_vns = df_vns['order_id'].to_list()

# # order_ids_vns.split('_')[1]
# # order_ids_vns = [order_id.split['_'][1] for order_id in order_ids_vns]
# order_ids_vns= [order_id.split('_')[1] for order_id in order_ids_vns]

# for index, row in df_aco.iterrows():
#     order_id_row = row['order_id']
#     if order_id_row not in order_ids_vns:
#         df_aco.drop(index, inplace=True )

# order_ids_aco_updated = df_aco['order_id'].to_list()
# count = Counter(order_ids_aco_updated).keys()



# df['DUETIME'] = 480
# print(df['DUETIME'])

df_vns.to_csv('/Users/marlenebuhl/Documents/University/thesis/coding_part/thesis_marlene/input_data/surve_mobility/orders/2020-07-07_orders_no_cancellations.csv')
# print('')