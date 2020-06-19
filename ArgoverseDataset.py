import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
import argoverse
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from tqdm import tqdm


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
        self.am = ArgoverseMap()

        self.axis_range = self.get_map_range(self.am)
        self.city_halluc_bbox_table, self.city_halluc_tableidx_to_laneid_map = self.am.build_hallucinated_lane_bbox_index()
        self.laneid_map = self.process_laneid_map()
        self.vector_map, self.extra_map = self.generate_vector_map()
        # am.draw_lane(city_halluc_tableidx_to_laneid_map['PIT']['494'], 'PIT')
        # self.save_vector_map(self.vector_map)

        self.last_observe = cfg['last_observe']
        ##set root_dir to the correct path to your dataset folder
        self.root_dir = cfg['data_locate']
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        self.map_feature = dict(PIT=[], MIA=[])
        self.city_name, self.center_xy, self.rotate_matrix = dict(), dict(), dict()

    def __len__(self):
        return len(self.afl)

    def __getitem__(self, index):
        # self.am.find_local_lane_polygons()
        self.trajectory, city_name, extra_fields = self.get_trajectory(index)
        traj_id = extra_fields['trajectory_id'][0]
        self.city_name.update({str(traj_id): city_name})
        center_xy = self.trajectory[self.last_observe-1][1]
        self.center_xy.update({str(traj_id): center_xy})
        trajectory_feature = (self.trajectory - np.array(center_xy).reshape(1, 1, 2)).reshape(-1, 4)
        rotate_matrix = get_rotate_matrix(trajectory_feature[self.last_observe, :])   # rotate coordinate
        self.rotate_matrix.update({str(traj_id): rotate_matrix})
        trajectory_feature = ((trajectory_feature.reshape(-1, 2)).dot(rotate_matrix.T)).reshape(-1, 4)
        trajectory_feature = self.normalize_coordinate(trajectory_feature, city_name)  # normalize to [-1, 1]

        self.traj_feature = torch.from_numpy(np.hstack((trajectory_feature,
                                                        extra_fields['TIMESTAMP'].reshape(-1, 1),
                                                        # extra_fields['OBJECT_TYPE'].reshape(-1, 1),
                                                        extra_fields['trajectory_id'].reshape(-1, 1)))).float()
        map_feature_dict = dict(PIT=[], MIA=[])
        for city in ['PIT', 'MIA']:
            for i in range(len(self.vector_map[city])):
                map_feature = (self.vector_map[city][i] -
                               np.array(center_xy).reshape(1, 1, 2)).reshape(-1, 2)
                map_feature = (map_feature.dot(rotate_matrix.T)).reshape(-1, 4)
                map_feature = self.normalize_coordinate(map_feature, city)
                tmp_tensor = torch.from_numpy(np.hstack((map_feature,
                                                         self.extra_map[city]['turn_direction'][i],
                                                         self.extra_map[city]['in_intersection'][i],
                                                         self.extra_map[city]['has_traffic_control'][i],
                                                         # self.extra_map[city]['OBJECT_TYPE'][i],
                                                         self.extra_map[city]['lane_id'][i])))
                map_feature_dict[city].append(tmp_tensor.float())
                # self.map_feature[city] = np.array(self.map_feature[city])
            self.map_feature[city] = map_feature_dict[city]
        self.map_feature['city_name'] = city_name
        return self.traj_feature, self.map_feature

    def get_trajectory(self, index):
        seq_path = self.afl.seq_list[index]
        data = self.afl.get(seq_path).seq_df
        data = data[data['OBJECT_TYPE'] == 'AGENT']
        extra_fields = dict(TIMESTAMP=[], OBJECT_TYPE=[], trajectory_id=[])
        polyline = []
        j = int(str(seq_path).split('/')[-1].split('.')[0])
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
            startpoint = np.array([xlast, ylast])
            endpoint = np.array([row['X'], row['Y']])
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
        return np.array(polyline), city_name, extra_fields

    def generate_vector_map(self):
        vector_map = {'PIT': [], 'MIA': []}
        extra_map = {'PIT': dict(OBJECT_TYPE=[], turn_direction=[], lane_id=[], in_intersection=[],
                                 has_traffic_control=[]),
                     'MIA': dict(OBJECT_TYPE=[], turn_direction=[], lane_id=[], in_intersection=[],
                                 has_traffic_control=[])}
        polyline = []
        # index = 1
        pbar = tqdm(total=17326)
        pbar.set_description("Generating Vector Map")
        for city_name in ['PIT', 'MIA']:
            for key in self.laneid_map[city_name]:
                pts = self.am.get_lane_segment_polygon(key, city_name)
                turn_str = self.am.get_lane_turn_direction(key, city_name)
                if turn_str == 'LEFT':
                    turn = -1
                elif turn_str == 'RIGHT':
                    turn = 1
                else:
                    turn = 0
                pts_len = pts.shape[0] // 2
                positive_pts = pts[:pts_len, :2]
                negative_pts = pts[pts_len:2 * pts_len, :2]
                polyline.clear()
                for i in range(pts_len - 1):
                    v1 = np.array([positive_pts[i], positive_pts[i + 1]])
                    v2 = np.array([negative_pts[pts_len - 1 - i], negative_pts[pts_len - i - 2]])
                    polyline.append(v1)
                    polyline.append(v2)
                    # extra_field['table_index'] = self.laneid_map[city_name][key]
                repeat_t = 2*(pts_len-1)
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
        for city_name in ['PIT', 'MIA']:
            poly = am.get_vector_map_lane_polygons(city_name)
            poly_modified = (np.vstack(poly))[:, :2]
            max_coordinate = np.max(poly_modified, axis=0)
            min_coordinate = np.min(poly_modified, axis=0)
            map_range[city_name].update({'max': max_coordinate})
            map_range[city_name].update({'min': min_coordinate})
        return map_range

    def normalize_coordinate(self, array, city_name):
        max_coordinate = self.axis_range[city_name]['max']
        min_coordinate = self.axis_range[city_name]['min']
        array = (10.*(array.reshape(-1, 2)) / (max_coordinate - min_coordinate)).reshape(-1,4)
        return array

    def save_vector_map(self, vector_map):
        save_path = "./data/vector_map/"
        for city_name in ['PIT', 'MIA']:
            tmp_map = np.vstack(vector_map[city_name]).reshape(-1, 4)
            np.save(save_path+city_name+"_vectormap", tmp_map)
