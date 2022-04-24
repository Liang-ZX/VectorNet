''' ArgoverseForecastDataset继承了torch.utils.data.Dataset，实现三个函数用于初始化和获取地图数据
    加载Argoverse HD map 和 Forecast 数据集，并将地图和轨迹数据进行向量化（vector map）归一化等处理
    由__getitem__函数将处理过的数据转为tensor并返回 '''

import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
import argoverse
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from common import *
import json
import pickle
import sys

## 根据轨迹点p0(x0,y0), p1(x1,y1)计算它们组成的向量的2x2旋转矩阵
def get_rotate_matrix(trajectory):
    x0, y0, x1, y1 = trajectory.flatten()
    vec1 = np.array([x1 - x0, y1 - y0])
    vec2 = np.array([0, 1])
    cosalpha = vec1.dot(vec2) / (np.sqrt(vec1.dot(vec1)) * 1 + 1e-5)
    sinalpha = np.sqrt(1 - cosalpha * cosalpha)
    if x1 - x0 < 0:
        sinalpha = -sinalpha
    rotate_matrix = np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
    return rotate_matrix


class ArgoverseForecastDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.am = ArgoverseMap()   # HD map in argoverse-api/map_files
        self.axis_range = self.get_map_range(self.am) # 获取整个城市的坐标范围，用于归一化坐标
        self.city_halluc_bbox_table, self.city_halluc_tableidx_to_laneid_map = self.am.build_hallucinated_lane_bbox_index() # 用于快速查询车道
        self.laneid_map = self.process_laneid_map()  # {'PIT': {9604854: '0'}, 'MIA': {9605252: '0'}}
        self.vector_map, self.extra_map = self.generate_vector_map() # get HD map and convert to vector, extra_map includes OBJECT_TYP, turn_direction, lane_id, in_intersection, has_traffic_control
        # am.draw_lane(city_halluc_tableidx_to_laneid_map['PIT']['494'], 'PIT')
        # self.save_vector_map(self.vector_map)
        
        self.last_observe = cfg['last_observe']
        ##set root_dir to the correct path to your dataset folder
        self.root_dir = cfg['data_locate']
        self.device = cfg['device']
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        # self.map_feature = dict(PIT=[], MIA=[])
        self.city_name, self.center_xy, self.rotate_matrix = dict(), dict(), dict()

    def __len__(self):
        return len(self.afl)

    def __getitem__(self, index):   # 迭代获取数据函数，在该函数中读取了trajectory数据，同时对坐标进行了一系列预处理，最后转换为归一化的轨迹和地图tensor
        # self.am.find_local_lane_polygons()
        self.trajectory, city_name, extra_fields = self.get_trajectory(index)   # 由索引获取一段轨迹，见图2021-10-11 21-36-01 的屏幕截图.png
        traj_id = extra_fields['trajectory_id'][0]  # 将xxx.csv中文件名作为scenario id，数据见data.txt，由于是同一段轨迹，所以id是一样的，所以我们取第0个
        self.city_name.update({str(traj_id): city_name})
        center_xy = self.trajectory[self.last_observe-1][1] # 将第last_observe-1个轨迹点作为中心点
        self.center_xy.update({str(traj_id): center_xy})    # 选取一个中心点，用于归一化处理,数据见data.txt, {'425': array([ 186.48895452, 1560.94612336])}
        trajectory_feature = (self.trajectory - np.array(center_xy).reshape(1, 1, 2)).reshape(-1, 4) # [[x1,y1,x2,y2],[x3,y3,x4,y4],...]
        rotate_matrix = get_rotate_matrix(trajectory_feature[self.last_observe, :])   # get rotate coordinate from last_observe vector
        self.rotate_matrix.update({str(traj_id): rotate_matrix})
        # 如果所有轨迹点都在一条直线上，那么旋转后的点都在y轴上
        trajectory_feature = ((trajectory_feature.reshape(-1, 2)).dot(rotate_matrix.T)).reshape(-1, 4) # 轨迹特征旋转并reshape
        # print('trajectory_feature before normalize :')
        # print(trajectory_feature)
        trajectory_feature = self.normalize_coordinate(trajectory_feature, city_name)  # 
        # np.savetxt('traj.txt',trajectory_feature,fmt='%0.8f')

         # 轨迹特征为6维[x1,y1,x2,y2,TIMESTAMP,trajectory_id]
        # self.traj_feature = torch.from_numpy(np.hstack((trajectory_feature,
        #                                                 extra_fields['TIMESTAMP'].reshape(-1, 1),
        #                                                 # extra_fields['OBJECT_TYPE'].reshape(-1, 1),
        #                                                 extra_fields['trajectory_id'].reshape(-1, 1)))).float()
        self.traj_feature = torch.from_numpy(trajectory_feature).float()

        # map_feature_dict = dict(PIT=[], MIA=[])
        # 地图特征为8维[v0x,v0y,v1x,v1y,turn_direction,in_intersection,has_traffic_control,lane_id]
        # 上面得到了self.center_xy和self.rotate_matrix，下面对每个点地图也需要做相应的去中心化和旋转
        self.map_feature = []
        # mf = []
        lane_ids = self.am.get_lane_ids_in_xy_bbox(center_xy[0], center_xy[1], city_name, 20)
        for id in lane_ids:
            index_str = self.laneid_map[city_name][id]
            i = int(index_str)
            vecmap_feature = (self.vector_map[city_name][i] - np.array(center_xy).reshape(1, 1, 2)).reshape(-1, 2) # 地图点去中心点
            vecmap_feature = (vecmap_feature.dot(rotate_matrix.T)).reshape(-1, 4) # 旋转并reshape
            vecmap_feature = self.normalize_coordinate(vecmap_feature, city_name) # 再归一化
            # mf.append(vecmap_feature)
            tmp_tensor = torch.from_numpy(np.hstack((vecmap_feature,
                                                     self.extra_map[city_name]['turn_direction'][i],
                                                     self.extra_map[city_name]['in_intersection'][i],
                                                     self.extra_map[city_name]['has_traffic_control'][i],
                                                     # self.extra_map[city_name]['OBJECT_TYPE'][i],
                                                     self.extra_map[city_name]['lane_id'][i])))
            self.map_feature.append(tmp_tensor)
        # map_length = len(self.map_feature)
        # if map_length > 32:
        #     self.map_feature = self.map_feature[:32]
        # elif map_length < 32:
        #     need_align = True
        #     while need_align:
        #         for i in range(map_length):
        #             self.map_feature.append(self.map_feature[i])
        #             if len(self.map_feature) == 32:
        #                 need_align = False
        #                 break
        

        # for city in ['PIT', 'MIA']:
        #     for i in range(len(self.vector_map[city])):
        #         map_feature = (self.vector_map[city][i] - np.array(center_xy).reshape(1, 1, 2)).reshape(-1, 2) # 地图点减去中心点作为map_feature
        #         map_feature = (map_feature.dot(rotate_matrix.T)).reshape(-1, 4) # 地图特征旋转并reshape
        #         map_feature = self.normalize_coordinate(map_feature, city)
        #         tmp_tensor = torch.from_numpy(np.hstack((map_feature,
        #                                                  self.extra_map[city]['turn_direction'][i],
        #                                                  self.extra_map[city]['in_intersection'][i],
        #                                                  self.extra_map[city]['has_traffic_control'][i],
        #                                                  # self.extra_map[city]['OBJECT_TYPE'][i],
        #                                                  self.extra_map[city]['lane_id'][i])))
        #         map_feature_dict[city].append(tmp_tensor.float())
        #         # self.map_feature[city] = np.array(self.map_feature[city])
        #     self.map_feature[city] = map_feature_dict[city]
        # self.map_feature['city_name'] = city_name

        # mapfeature = np.vstack(mf)
        # np.savetxt('map.txt',mapfeature,fmt='%0.8f')
        # sys.exit()
        return self.traj_feature, self.map_feature  # 返回的是一条5s轨迹向量(49,6)和中心点周围的n个地图向量(n,18,8)

    def get_trajectory(self, index):
        seq_path = self.afl.seq_list[index]
        data = self.afl.get(seq_path).seq_df    # Get the dataframe for the current sequence. 见docs/data.txt
        data = data[data['OBJECT_TYPE'] == 'AGENT'] # will get AGENT traject, 取出所有agent的轨迹
        extra_fields = dict(TIMESTAMP=[], OBJECT_TYPE=[], trajectory_id=[])
        polyline = []
        j = int(str(seq_path).split('/')[-1].split('.')[0]) # forecating sequence 123.cvs文件名去掉后缀(一串数字)
        flag = True
        city_name = ''
        for _, row in data.iterrows():
            if flag:
                xlast = row['X']
                ylast = row['Y']
                tlast = row['TIMESTAMP']
                city_name = row['CITY_NAME']
                flag = False
                continue
            startpoint = np.array([xlast, ylast])       # 相邻点组成向量
            endpoint = np.array([row['X'], row['Y']])
            # plt.annotate('', xy=(endpoint[0],endpoint[1]),xytext=(startpoint[0],startpoint[1]),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
            xlast = row['X']
            ylast = row['Y']
            extra_fields['TIMESTAMP'].append(tlast)
            extra_fields['OBJECT_TYPE'].append(0)  # 'AGENT'
            extra_fields['trajectory_id'].append(j)  # 'AGENT'
            tlast = row['TIMESTAMP']
            polyline.append([startpoint, endpoint])
        extra_fields['TIMESTAMP'] = np.array(extra_fields['TIMESTAMP'])
        extra_fields['TIMESTAMP'] -= np.min(extra_fields['TIMESTAMP'])  # adjust time stamp
        extra_fields['OBJECT_TYPE'] = np.array(extra_fields['OBJECT_TYPE'])
        extra_fields['trajectory_id'] = np.array(extra_fields['trajectory_id'])
        # plt.show()
        return np.array(polyline), city_name, extra_fields

    def generate_vector_map(self):  # 读取HD map并转换成vector，返回vector map和由其他信息组成的extra_map
        vector_map = {'PIT': [], 'MIA': []}
        extra_map = {'PIT': dict(OBJECT_TYPE=[], turn_direction=[], lane_id=[], in_intersection=[],
                                 has_traffic_control=[]),
                     'MIA': dict(OBJECT_TYPE=[], turn_direction=[], lane_id=[], in_intersection=[],
                                 has_traffic_control=[])}
        polyline = []
        # index = 1
        pbar = tqdm(total=17326)    # 进度条
        pbar.set_description("Generating Vector Map")

        # city_name = 'MIA'
        # for i in range(1):
        #     key = 9624155 + i
        #     pts = self.am.get_lane_segment_polygon(key, city_name)
        #     pts = pts[:,:2]
        #     print(pts)
        #     x1 = pts[:,0]
        #     y1 = pts[:,1]
        #     plt.plot(x1, y1,'ro')
        #     pts_len = pts.shape[0] // 2                     # 21 // 2 返回10
        #     positive_pts = pts[:pts_len, :2]                # 车道左边界(x,y)坐标
        #     negative_pts = pts[pts_len:2 * pts_len, :2]     # 右边界
        #     for i in range(pts_len - 1):
        #         v1 = np.array([positive_pts[i], positive_pts[i + 1]])   # 车道左边界向量
        #         v2 = np.array([negative_pts[pts_len - 1 - i], negative_pts[pts_len - i - 2]])   # 右边界向量
        #         plt.annotate('', xy=(positive_pts[i+1][0],positive_pts[i+1][1]),xytext=(positive_pts[i][0],positive_pts[i][1]),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        #         plt.annotate('', xy=(negative_pts[pts_len - i - 2][0],negative_pts[pts_len - i - 2][1]),xytext=(negative_pts[pts_len - 1 - i][0],negative_pts[pts_len - 1 - i][1]),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        # plt.show()

        for city_name in ['PIT', 'MIA']:
            for key in self.laneid_map[city_name]: # lane id
                # 由lane_id (key) 和 city_name 返回的pts是由21个三维坐标点(x,y,z)组成的一个闭合车道（第一个点和最后一个点重合）
                pts = self.am.get_lane_segment_polygon(key, city_name)  # get lane boundries sample points, stitch them as vector (specified in the paper)
                turn_str = self.am.get_lane_turn_direction(key, city_name)
                if turn_str == 'LEFT':
                    turn = -1
                elif turn_str == 'RIGHT':
                    turn = 1
                else:
                    turn = 0
                pts_len = pts.shape[0] // 2                     # 21 // 2 返回10
                positive_pts = pts[:pts_len, :2]                # 车道左边界(x,y)坐标
                negative_pts = pts[pts_len:2 * pts_len, :2]     # 右边界
                # if city_name == 'PIT':
                #     plt.plot(pts[:pts_len, 0], pts[:pts_len, 1])
                #     plt.plot(pts[pts_len:2 * pts_len, 0], pts[pts_len:2 * pts_len, 1])
                polyline.clear()
                for i in range(pts_len - 1):
                    v1 = np.array([positive_pts[i], positive_pts[i + 1]])   # 车道左边界向量，二维向量只用xy坐标
                    v2 = np.array([negative_pts[pts_len - 1 - i], negative_pts[pts_len - i - 2]])   # 右边界向量
                    polyline.append(v1)
                    polyline.append(v2)
                    # extra_field['table_index'] = self.laneid_map[city_name][key]
                repeat_t = 2*(pts_len-1)
                # 最后得到的polyline是18维的，每一维是用两个点表示的向量，转成np.array再加到vector_map
                vector_map[city_name].append(np.array(polyline).copy())
                extra_map[city_name]['turn_direction'].append(np.repeat(turn, repeat_t, axis=0).reshape(-1, 1))
                extra_map[city_name]['OBJECT_TYPE'].append(np.repeat(-1, repeat_t, axis=0).reshape(-1, 1)) #HD Map
                extra_map[city_name]['lane_id'].append(np.repeat(int(key), repeat_t, axis=0).reshape(-1, 1))
                extra_map[city_name]['in_intersection'].append(np.repeat(
                    1 * self.am.lane_is_in_intersection(key, city_name), repeat_t, axis=0).reshape(-1, 1))
                extra_map[city_name]['has_traffic_control'].append(np.repeat(
                    1 * self.am.lane_has_traffic_control_measure(key, city_name), repeat_t, axis=0).reshape(-1, 1))
                # if index > 10:
                #     break
                # index = index + 1
                pbar.update(1)
        pbar.close()
        # plt.show()
        # mylog = open('extra_map.txt', mode = 'a',encoding='utf-8')
        # print(extra_map, file=mylog)

        print("Generate Vector Map Successfully!")
        return vector_map, extra_map #vector_map:list

    def process_laneid_map(self):
        laneid_map = {}
        tmp_map = {}
        tmp1_map = {}
        for key in self.city_halluc_tableidx_to_laneid_map['PIT']:
            tmp_map[self.city_halluc_tableidx_to_laneid_map['PIT'][key]] = key
        laneid_map['PIT'] = tmp_map
        for key in self.city_halluc_tableidx_to_laneid_map['MIA']:
            tmp1_map[self.city_halluc_tableidx_to_laneid_map['MIA'][key]] = key
        laneid_map['MIA'] = tmp1_map
        return laneid_map

    def get_map_range(self, am):
        map_range = dict(PIT={}, MIA={})
        for city_name in ['PIT', 'MIA']:                        # 匹兹堡，迈阿密
            poly = am.get_vector_map_lane_polygons(city_name)   # Get list of lane polygons for a specified city
            poly_modified = (np.vstack(poly))[:, :2]            # 所有地图数据垂直排列，取前两列xy
            max_coordinate = np.max(poly_modified, axis=0)      # xy轴的最大值和最小值
            min_coordinate = np.min(poly_modified, axis=0)
            map_range[city_name].update({'max': max_coordinate})
            map_range[city_name].update({'min': min_coordinate})
            print(city_name + ' map range :')
            print(max_coordinate)
            print(min_coordinate)
        return map_range

    def normalize_coordinate(self, array, city_name):
        max_coordinate = self.axis_range[city_name]['max']
        min_coordinate = self.axis_range[city_name]['min']
        array = (100.*(array.reshape(-1, 2)) / (max_coordinate - min_coordinate)).reshape(-1,4)
        return array

    def save_vector_map(self, vector_map):
        save_path = "./data/vector_map/"
        for city_name in ['PIT', 'MIA']:
            tmp_map = np.vstack(vector_map[city_name]).reshape(-1, 4)
            np.save(save_path+city_name+"_vectormap", tmp_map)
