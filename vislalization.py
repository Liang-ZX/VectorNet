#!/usr/bin/python

import torch
import torch.nn
import argoverse
import os
import matplotlib.pyplot as plt
import numpy as np

from argoverse.map_representation.map_api import ArgoverseMap

def show_argoverse_map():
	am = ArgoverseMap()

	laneid_map = {}
	city_halluc_bbox_table, city_halluc_tableidx_to_laneid_map = am.build_hallucinated_lane_bbox_index()
	for key in city_halluc_tableidx_to_laneid_map['PIT']:
	    laneid_map[city_halluc_tableidx_to_laneid_map['PIT'][key]] = key

	# print(laneid_map[9618489])
	# am.draw_lane(9618489,'PIT')

	for key in laneid_map:
	    center_line = am.get_lane_segment_centerline(key, 'PIT')
	    if np.min(center_line, axis=0)[0]>2500 and np.max(center_line, axis=0)[0] < 3000:
	        am.draw_lane(key,'PIT')

	plt.show()

def print_forecasting_data():
	from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
	afl = ArgoverseForecastingLoader("/home/tangx2/storage/projects/git/argoverse-api/train/data_10")
	seq_path = afl.seq_list[7]
	data = afl.get(seq_path).seq_df
	data = data[data['OBJECT_TYPE'] == 'AGENT']

	print(data.head())

	afl.seq_list[7]
	plt.figure()
	plt.xlim(1945, 2000)
	plt.ylim(635, 685)
	for i in range(vectormap.shape[0]):
	    if(vectormap[i][0] > 1945 and vectormap[i][0] < 2000 and vectormap[i][1]>635 and vectormap[i][1]<685):
	        plt.arrow(vectormap[i][0], vectormap[i][1], vectormap[i][2]-vectormap[i][0], vectormap[i][3]-vectormap[i][1],
	                 length_includes_head = True,head_width = 0.25,head_length = 0.5)

	plt.show()

def main():
	print_forecasting_data()

if __name__ == '__main__':
	main()

