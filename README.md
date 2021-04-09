# VectorNet Re-implementation

This is the unofficial **pytorch** implementation of CVPR2020 paper *"VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation"*. (And it's a part of test of the summer camp 2020 organized by IIIS, Tsinghua University.)

1. 运行环境

   python 3.7, Pytorch1.1.0, torchvision0.3.0, cuda9.0 

2. 文件说明

   ----- VectorNet

   +--- ArgoverseDataset.py	数据集读取、预处理、转换为tensor

   +--- subgraph_net.py		   polyline subgraph相关类实现

   +--- gnn.py							 带Attention机制的GCN，因为图是全连接，所以没有用dgl

   +--- vectornet.py				   把subgraph和GNN合并起来的model，loss计算

   +--- train.py						    网络训练入口，会保存checkpoint

   +--- test.py							 网络测试入口，同时实现了评估函数，会保存inference结果

   +--- Visualization.ipynb		可视化vectorize的HD map

3. 运行准备

   - 安装[argoverse-api](https://github.com/argoai/argoverse-api)且按照说明，将HD map数据放置到指定位置
   - 下载forecast数据集，将train.py和test.py中```cfg['data_locate']```修改为解压位置

4. 代码函数解读

   - ArgoverseDataset.py 

     定义了类```class ArgoverseForecastDataset(torch.utils.data.Dataset)```

     - ```def __init__(self, cfg)``` 类初始化，主要步骤有

       ```python
       self.axis_range = self.get_map_range(self.am) #用于normalize坐标
       self.city_halluc_bbox_table, self.city_halluc_tableidx_to_laneid_map = self.am.build_hallucinated_lane_bbox_index()
       self.vector_map, self.extra_map = self.generate_vector_map()
       ```

       调用argoverse api读取HD map数据，重点是```generate_vector_map```函数

     - ```def generate_vector_map(self)``` 读取HD map并转换成vector

       利用argoverse api的```get_lane_segment_polygon(key, city_name)``` 获取道路边沿的采样点，以论文指定的vector的方式拼接，该api是得到polygon，而我们只要两个边沿，因此做了一些处理

       同时将相关semantic label获取，返回至extra_map，待后续组装进vector内

     - ```def __getitem__(self, index)``` 迭代获取数据函数，在该函数中读取了trajectory数据，同时对坐标进行了一系列预处理，最后转换为tensor

       获取trajectory同样利用argoverse api，数据预处理主要分为3个步骤

       (1)平移坐标使last_observe移到中心 

       (2)rotate利用齐次坐标旋转矩阵实现，夹角利用向量内积获得 

       (3)normalize这里通过线性变换把坐标normalize到一定范围，这里认为last_observe的位置就是数据集分布的中心，即
       $$
       x = \frac{x}{max-min}
       $$
       
      - ```__getitem__```返回
     
         ```python
              self.traj_feature, self.map_feature
         ```
         其中```self.traj_feature``` 是$N\times feature$ 维的tensor指示轨迹polyline的vector集合 ```self.map_feature``` 是一个有三个key的dict，			```map_feature['PIT']和map_feature['MIA']``` 是list，分别是两座城市道路的polyline的list，即list的每一个元素是一个$N\times feature$ 维的tensor，指示一条道路的polyline，```map_feature['city_name']```保存该trajectory所在的城市
         ```def get_trajectory(self, index)``` 与 ```generate_vector_map``` 类似，区别在于trajectory是针对timestamp进行轨迹拼接，同时需要将timestamp装入向量中作为semantic label的信息



   - subgraph_net.py

     定义了类```class SubgraphNet(nn.Module) ```和 ```class SubgraphNet_Layer(nn.Module)```

     - 类```class SubgraphNet_Layer``` 

       输入：$N\times feature$ 维的单polyline tensor

       输出：$N\times (feature+global\ feature)$ 维的单polyline tensor

       实现了单层的SubgraphNet，按照文章叙述，*encoder*是一个MLP，具体由一个全连接层、一个**layer_norm** 和一个RELU激发层组成，随后是*max_pool*提取全局信息，最后*concatenate*将信息整合，与Point R-CNN相似

     - 类 ```class SubgraphNet```

       输入：$N\times feature$ 维的单polyline tensor

       输出：$1\times (feature+global\ feature)$ 维的单polyline tensor

       将 **3** 层SubgraphNet_Layer组合，最后*max_pool*提取代表性信息



   - gnn.py

     定义了类```class GraphAttentionNet(nn.Module)```

     - 类```class GraphAttentionNet```

       输入：$K\times (feature+global\ feature)$ 维的全图特征信息

       输出：$K\times value\ dims$ 维的传播后全图特征信息

       因为在本论文中，将邻接矩阵定义为全连接矩阵，因此没有建图实现消息传播的必要性。Attention机制在本类中加以实现，公式即为
       $$
       GNN(P)=softmax(P_QP_K^T)P_V
       $$
       注意：这里进行的都是矩阵计算。$P_Q$是查询，$P_K$是key，$P_V$是值，*softmax*一步是获得各value的权重

       具体的实现参考了论文**Attention is All you need**



   - vectornet.py

     定义了类```class VectorNet(nn.Module)```

     - 类```class VectorNet``` 本类的 *forward* 分 *train* 和 *evaluate* 两种情况

       输入：trajectory_batch, mapfeature_batch

       输出：train时输出loss，evaluate时输出预测结果predictions和真值label

        - 由于不同道路的polyline采样点数不同，因此在dataset数据读取时把它放入了list中，因此在本类中会首先完成对数据的拆包
        - 然后构造两个SubgraphNet类，```traj_subgraphnet```，和```map_subgraphnet```将不同polyline的信息，都处理为$1\times (feature+global\ feature)$ 维的polyline信息，然后*concatenate*起来
        - 此后会进行L2 normalize以有效训练后面的GNN，正则化后直接传入GNN，并得到传播后的vector信息 $1\times value\ dims$ 维，decoder使用了MLP与subgraph_net参数相似，但多加了一层全连接网络以生成回归坐标
        - 如果是train则使用torch.nn.MSEloss计算损失，可以证明在误差服从标准高斯分布时，Gaussian Negative Likelihood Loss就是MSEloss，它们本质上是等价的。如果是evaluate则把prediction和label一起输出，在test.py中实现Average Displacement Error的计算



   - train.py
  
     网络训练入口
  
     - ```def main()```
  
       首先初始化一些参数，为代码简便，这里把配置(cfg)直接编码在代码中，更合适的做法应是利用 argparse 通过命令行传入。然后实例化dataset，利用dataloader打包为minibatch，初始化model，设置优化器，和步长自调节器
  
       另外这里使用*tensorboard*可视化损失，文件保存在 ./run/文件夹下，因此需要初始化SummaryWriter
  
     - ```def do_train(model, cfg, train_loader, optimizer, scheduler, writer)```
  
       较为常见的主训练循环，每 **5** 个epoch调节一次步长，每10个epoch保存一次模型参数，训练结束保存一次模型参数，输出每2个iteration(minibatch)输出一次信息，采用logger保存日志文件
  
     
  
   - test.py
  
     网络推断入口
  
     - ```def main()```
  
       与train.py几乎相同，注意cfg['model_path']模型参数文件路径和cfg['save_path']推理结果存储路径两个参数
  
     - ```def inference(model, cfg, val_loader)```
  
       较do_train有所简化，因为无需再处理vector_map数据，已经被编码进网络里（*只使用了一层的GNN*），将输出的result和label用list保存起来，调用```evaluate()```函数计算**ADE**指标
  
     - ```def evaluate(dataset, predictions, labels)```
  
       传入dataset是因为需要把预处理过的数据，变换回原始坐标，即先反归一化，然后逆向旋转，最后平移，ADE loss即是预测点和真值点间欧氏距离的平均，inference的结果保存在路径cfg['save_path']下
  
   
  
5. 一些可视化的结果(详见visualization.ipynb)

   - loss 收敛(150组数据，训练了25个epoch，adadelta优化器，有点过拟合)
   ![img1](https://user-images.githubusercontent.com/42173433/112776253-bbb77a00-9071-11eb-8125-3f3c53b117c5.png)  
   ![img2](https://user-images.githubusercontent.com/42173433/112776261-c3771e80-9071-11eb-8f80-70280af320b1.png)
   - baseline的结果(150组数据，训练了10个epoch，9步预测)
   ![img3](https://user-images.githubusercontent.com/42173433/112776280-cf62e080-9071-11eb-92c3-92430df63a11.png)
   - 地图矢量化  
   ![img1](https://user-images.githubusercontent.com/42173433/112776359-105af500-9072-11eb-82c4-1ebf6790a5a0.png)  
   ![img4](https://user-images.githubusercontent.com/42173433/112776373-1650d600-9072-11eb-8475-db1dce02a632.png)
   - 轨迹预测(蓝色的是label，红色是预测，十字路口场景呈现回归现象)  
   ![img2](https://user-images.githubusercontent.com/42173433/112776385-1c46b700-9072-11eb-82ea-12822871e6d1.png)

     
