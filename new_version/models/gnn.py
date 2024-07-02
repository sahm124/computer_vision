from functools import partial
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def instance_normalization(features):
    mean, variance = tf.nn.moments(features, [0], name='IN_stats', keepdims=True)
    features = tf.nn.batch_normalization(features, mean, variance, None, None, 1e-12, name='IN_apply')
    return features

normalization_fn_dict = {
    'fused_BN_center': tf.keras.layers.BatchNormalization,
    'BN': partial(tf.keras.layers.BatchNormalization, center=False),
    'BN_center': tf.keras.layers.BatchNormalization,
    'IN': instance_normalization,
    'NONE': None
}

activation_fn_dict = {
    'ReLU': tf.nn.relu,
    'ReLU6': tf.nn.relu6,
    'LeakyReLU': partial(tf.nn.leaky_relu, alpha=0.01),
    'ELU': tf.nn.elu,
    'NONE': None,
    'Sigmoid': tf.nn.sigmoid,
    'Tanh': tf.nn.tanh,
}

def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
    is_logits=False, num_layer=4, normalization_type="fused_BN_center",
    activation_type='ReLU'):
    assert len(sv.shape) == 2
    assert len(Ks) == num_layer-1
    if is_logits:
        features = sv
        for i in range(num_layer-1):
            features = tf.keras.layers.Dense(Ks[i],
                activation=activation_fn_dict[activation_type],
                kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
            if normalization_fn_dict[normalization_type] is not None:
                features = normalization_fn_dict[normalization_type]()(features)
        features = tf.keras.layers.Dense(num_classes,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
    else:
        features = sv
        for i in range(num_layer-1):
            features = tf.keras.layers.Dense(Ks[i],
                activation=activation_fn_dict[activation_type],
                kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
            if normalization_fn_dict[normalization_type] is not None:
                features = normalization_fn_dict[normalization_type]()(features)
        features = tf.keras.layers.Dense(num_classes,
            activation=activation_fn_dict[activation_type],
            kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
        if normalization_fn_dict[normalization_type] is not None:
            features = normalization_fn_dict[normalization_type]()(features)
    if mask is not None:
        features = features * mask
    return features

def multi_layer_neural_network_fn(features, Ks=(64, 32, 64), is_logits=False,
    normalization_type="fused_BN_center", activation_type='ReLU'):
    assert len(features.shape) == 2
    if is_logits:
        for i in range(len(Ks)-1):
            features = tf.keras.layers.Dense(Ks[i],
                activation=activation_fn_dict[activation_type],
                kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
            if normalization_fn_dict[normalization_type] is not None:
                features = normalization_fn_dict[normalization_type]()(features)
        features = tf.keras.layers.Dense(Ks[-1],
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
    else:
        for i in range(len(Ks)):
            features = tf.keras.layers.Dense(Ks[i],
                activation=activation_fn_dict[activation_type],
                kernel_initializer=tf.keras.initializers.GlorotUniform())(features)
            if normalization_fn_dict[normalization_type] is not None:
                features = normalization_fn_dict[normalization_type]()(features)
    return features

def graph_scatter_max_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_max(point_features,
        point_centers, num_centers, name='scatter_max')
    return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_sum(point_features,
        point_centers, num_centers, name='scatter_sum')
    return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_mean(point_features,
        point_centers, num_centers, name='scatter_mean')
    return aggregated

class ClassAwarePredictor(layers.Layer):
    def __init__(self, cls_fn, loc_fn):
        super(ClassAwarePredictor, self).__init__()
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        box_encodings_list = []
        with tf.name_scope('predictor'):
            with tf.name_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            with tf.name_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.name_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features, num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class ClassAwareSeparatedPredictor(layers.Layer):
    def __init__(self, cls_fn, loc_fn):
        super(ClassAwareSeparatedPredictor, self).__init__()
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        box_encodings_list = []
        with tf.name_scope('predictor'):
            with tf.name_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            features_splits = tf.split(features, num_classes, axis=-1)
            with tf.name_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.name_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features_splits[class_idx],
                            num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class GatherLayer(layers.Layer):
    def call(self, inputs):
        params, indices = inputs
        return tf.gather(params, indices)

class PointSetPooling(layers.Layer):
    def __init__(self,
        point_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        output_fn=multi_layer_neural_network_fn):
        super(PointSetPooling, self).__init__()
        self._point_feature_fn = point_feature_fn
        self._aggregation_fn = aggregation_fn
        self._output_fn = output_fn
        self._gather = GatherLayer()

    def call(self, inputs):
        point_features, point_coordinates, keypoint_indices, set_indices, point_MLP_depth_list, point_MLP_normalization_type, point_MLP_activation_type, output_MLP_depth_list, output_MLP_normalization_type, output_MLP_activation_type = inputs
        point_set_features = self._gather([point_features, set_indices[:,0]])
        point_set_coordinates = self._gather([point_coordinates, set_indices[:,0]])
        point_set_keypoint_indices = self._gather([keypoint_indices, set_indices[:, 1]])
        point_set_keypoint_coordinates = self._gather([point_coordinates, point_set_keypoint_indices[:,0]])
        point_set_coordinates = point_set_coordinates - point_set_keypoint_coordinates
        point_set_features = tf.concat([point_set_features, point_set_coordinates], axis=-1)
        with tf.name_scope('extract_vertex_features'):
            extracted_point_features = self._point_feature_fn(
                point_set_features,
                Ks=point_MLP_depth_list, is_logits=False,
                normalization_type=point_MLP_normalization_type,
                activation_type=point_MLP_activation_type)
            set_features = self._aggregation_fn(
                extracted_point_features, set_indices[:, 1],
                tf.shape(keypoint_indices)[0])
        with tf.name_scope('combined_features'):
            set_features = self._output_fn(set_features,
                Ks=output_MLP_depth_list, is_logits=False,
                normalization_type=output_MLP_normalization_type,
                activation_type=output_MLP_activation_type)
        return set_features

    def apply_regular(self,
        point_features,
        point_coordinates,
        keypoint_indices,
        set_indices,
        point_MLP_depth_list=None,
        point_MLP_normalization_type='fused_BN_center',
        point_MLP_activation_type='ReLU',
        output_MLP_depth_list=None,
        output_MLP_normalization_type='fused_BN_center',
        output_MLP_activation_type='ReLU'):
        return self.call([point_features, point_coordinates, keypoint_indices, set_indices, point_MLP_depth_list, point_MLP_normalization_type, point_MLP_activation_type, output_MLP_depth_list, output_MLP_normalization_type, output_MLP_activation_type])

class GraphNetAutoCenter(layers.Layer):
    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        super(GraphNetAutoCenter, self).__init__()
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn
        self._gather = GatherLayer()

    def call(self, inputs):
        input_vertex_features, input_vertex_coordinates, NOT_USED, edges, edge_MLP_depth_list, edge_MLP_normalization_type, edge_MLP_activation_type, update_MLP_depth_list, update_MLP_normalization_type, update_MLP_activation_type, auto_offset, auto_offset_MLP_depth_list, auto_offset_MLP_normalization_type, auto_offset_MLP_feature_activation_type = inputs
        s_vertex_features = self._gather([input_vertex_features, edges[:,0]])
        s_vertex_coordinates = self._gather([input_vertex_coordinates, edges[:,0]])
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        d_vertex_coordinates = self._gather([input_vertex_coordinates, edges[:, 1]])
        edge_features = tf.concat(
            [s_vertex_features, s_vertex_coordinates - d_vertex_coordinates],
             axis=-1)
        with tf.name_scope('extract_vertex_features'):
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        with tf.name_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        output_vertex_features = update_features + input_vertex_features
        return output_vertex_features

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type='ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type='ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type='ReLU'):
        return self.call([input_vertex_features, input_vertex_coordinates, NOT_USED, edges, edge_MLP_depth_list, edge_MLP_normalization_type, edge_MLP_activation_type, update_MLP_depth_list, update_MLP_normalization_type, update_MLP_activation_type, auto_offset, auto_offset_MLP_depth_list, auto_offset_MLP_normalization_type, auto_offset_MLP_feature_activation_type])
