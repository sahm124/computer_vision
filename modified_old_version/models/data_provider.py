# data_provider.py

import numpy as np
from models.graph_gen import get_graph_generate_fn
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
from models import preprocess
from dataset.kitti_dataset import KittiDataset

def fetch_data(frame_idx, dataset, config, train_config, aug_fn, sampler=None):
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx, config['downsample_by_voxel_size'])
    box_label_list = dataset.get_label(frame_idx)
    if 'crop_aug' in train_config:
        cam_rgb_points, box_label_list = sampler.crop_aug(cam_rgb_points, box_label_list,
                                                          sample_rate=train_config['crop_aug']['sample_rate'],
                                                          parser_kwargs=train_config['crop_aug']['parser_kwargs'])
    cam_rgb_points, box_label_list = aug_fn(cam_rgb_points, box_label_list)
    graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])
    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)), cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]], np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    last_layer_graph_level = config['model_kwargs']['layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]
    if config['label_method'] == 'yaw':
        cls_labels, boxes_3d, valid_boxes, label_map = dataset.assign_classaware_label_to_points(
            box_label_list, last_layer_points_xyz, expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    elif config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map = dataset.assign_classaware_car_label_to_points(
            box_label_list, last_layer_points_xyz, expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    elif config['label_method'] == 'Pedestrian_and_Cyclist':
        cls_labels, boxes_3d, valid_boxes, label_map = dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz, expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz, boxes_3d, label_map)
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return input_v, vertex_coord_list, keypoint_indices_list, edges_list, cls_labels, encoded_boxes, valid_boxes

class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, dataset should be
    reloaded.
    """
    def __init__(self, fetch_data, batch_data, load_dataset_to_mem=True, load_dataset_every_N_time=1,
                 capacity=1, num_workers=1, preload_list=[], async_load_rate=1.0, result_pool_limit=10000):
        self._fetch_data = fetch_data
        self._batch_data = batch_data
        self._buffer = {}
        self._results = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
        self._worker_pool = multiprocessing.Pool(processes=num_workers)
        self._preload_list = preload_list
        self._async_load_rate = async_load_rate
        self._result_pool_limit = result_pool_limit
        if len(self._preload_list) > 0:
            self.preload(self._preload_list)

    def preload(self, frame_idx_list):
        """async load dataset into memory."""
        for frame_idx in frame_idx_list:
            result = self._worker_pool.apply_async(self._fetch_data, (frame_idx,))
            self._results[frame_idx] = result

    def async_load(self, frame_idx):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            del self._results[frame_idx]
        else:
            data = self._fetch_data(frame_idx)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                result = self._worker_pool.apply_async(self._fetch_data, (frame_idx,))
                self._results[frame_idx] = result
        return data

    def provide(self, frame_idx):
        if self._load_dataset_to_mem:
            if self._load_every_N_time >= 1:
                extend_frame_idx = frame_idx + np.random.choice(self._capacity) * NUM_TEST_SAMPLE
                if extend_frame_idx not in self._buffer:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                if ctr == self._load_every_N_time:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                self._buffer[extend_frame_idx] = (data, ctr + 1)
                return data
            else:
                # do not buffer
                return self.async_load(frame_idx)
        else:
            return self._fetch_data(frame_idx)

    def provide_batch(self, frame_idx_list):
        batch_list = []
        for frame_idx in frame_idx_list:
            batch_list.append(self.provide(frame_idx))
        return self._batch_data(batch_list)
