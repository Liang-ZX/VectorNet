#!/usr/bin/python

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import os

##set root_dir to the correct path to your dataset folder
root_dir = '/home/tangx2/storage/projects/git/argoverse-api/train/data_10/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:',len(afl))

argoverse_forecasting_data = afl[0]
print(argoverse_forecasting_data.track_id_list)

# seq_path = f"{root_dir}/26.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)
# seq_path = f"{root_dir}/261.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)

# for filename in os.listdir(root_dir):
#     print(filename)
#     viz_sequence(afl.get(root_dir+filename).seq_df, show=True)


avm = ArgoverseMap()

# obs_len = 20

# index = 2
# seq_path = afl.seq_list[index]
# agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
# candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[index].city, viz=True)

# index = 3
# seq_path = afl.seq_list[index]
# agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
# candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[index].city, viz=True)

index = 2
seq_path = afl.seq_list[index]
agent_traj = afl.get(seq_path).agent_traj
lane_direction = avm.get_lane_direction(agent_traj[0], afl[index].city, visualize=True)

index = 3
seq_path = afl.seq_list[index]
agent_traj = afl.get(seq_path).agent_traj
lane_direction = avm.get_lane_direction(agent_traj[0], afl[index].city, visualize=True)

plt.show()