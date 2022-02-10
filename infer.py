import argparse
import os
import pickle
import time
from main import Config, read_data2, MapInfo
from geo import Map, GeoPoint
import tensorflow as tf
from trajmodel import TrajModel
import pandas as pd


class RoutePlanner(object):
    def __init__(self, config, ckpt_path):
        config.batch_size = 1
        routes, train, valid, test = read_data2(config.train_path, config.test_path, config.data_size, config.max_seq_len)

        GeoPoint.AREA_LAT = 41.15  # the latitude of the testing area. In fact, any value is ok in this problem.
        roadnet = Map()
        roadnet.open(config.map_path)

        # set config
        config.set_config(routes, roadnet)
        config.printf()

        # extract map info
        mapInfo = MapInfo(roadnet, config)

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        model_scope = "Model"
        with tf.name_scope("Train"):
            with tf.variable_scope(model_scope, reuse=None, initializer=initializer):
                self.model = TrajModel(not config.trace_hid_layer, config, train, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = tf.InteractiveSession(config=sess_config)

        print('try loading ' + ckpt_path)
        self.model.saver.restore(self.session, ckpt_path)
        print("restoring model trainable params from %s successfully" % ckpt_path)

    def planning(self, o, d, max_seq_len=75):
        route = [o]
        step = 2
        while step < max_seq_len and route[-1] != d:
            data = [route + [d]]
            vals = self.model.predict(self.session, data)
            route.append(vals['output'][-1])
            step = len(route)
        return route


def metric_out(ratios):
    df = pd.DataFrame(ratios, columns=['dis', 'gen', 'real', 'con'])
    dis = df['dis'].tolist()
    gen = df['gen'].tolist()
    real = df['real'].tolist()
    con = df['con'].tolist()

    p = sum(dis) * 1.0 / sum(gen)
    r = sum(dis) * 1.0 / sum(real)
    f1 = 2 * p * r / (p + r)
    jac = sum(dis) * 1.0 / sum(con)

    return [round(p, 3), round(r, 3), round(f1, 3), round(jac, 3)]


def f1_metric(pred, ground, segs_info):
    pred = set(pred)
    ground = set(ground)
    disjunction = pred & ground
    conjunction = pred | ground
    res = [len(disjunction), len(pred), len(ground), len(conjunction)]
    weighted_res = [0, 0, 0, 0]
    for edge in disjunction:
        length = segs_info[edge]['length']
        weighted_res[0] += length
    for edge in pred:
        length = segs_info[edge]['length']
        weighted_res[1] += length
    for edge in ground:
        length = segs_info[edge]['length']
        weighted_res[2] += length
    for edge in conjunction:
        length = segs_info[edge]['length']
        weighted_res[3] += length
    return res, weighted_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=str, default="0")
    parser.add_argument('--model', type=str, default="./ckpt/model.ckpt")
    parser.add_argument('-len_class', type=str, default="short")
    parser.add_argument('--test_data', type=str, default="/Users/tianwei/Projects/AttnRP/data/cd/lengroup_seqs")
    parser.add_argument('--road_file', type=str, default="/Users/tianwei/Projects/AttnRP/data/cd/road_graph")
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    config = Config("config")
    ckpt_path = opt.model
    planner = RoutePlanner(config, ckpt_path)

    # route = planner.planning(5772, 6046)
    # print(route)

    G = pickle.load(open(opt.road_file, "rb"))
    test = pickle.load(open(opt.test_data, "rb"))
    test = test[opt.len_class]

    metric = [0, 0, 0, 0]
    metric_weighted = [0, 0, 0, 0]
    unreachable = 0
    total = 0

    start_time = time.time()
    for path in test:
        if len(path) < 3:
            continue
        pred = planner.planning(path[0], path[-1])
        total += 1
        if pred[-1] != path[-1]:
            unreachable += 1
        res, res_weighted = f1_metric(set(pred), set(path), G.nodes)
        metric[0] += res[0]
        metric[1] += res[1]
        metric[2] += res[2]
        metric[3] += res[3]
        metric_weighted[0] += res_weighted[0]
        metric_weighted[1] += res_weighted[1]
        metric_weighted[2] += res_weighted[2]
        metric_weighted[3] += res_weighted[3]

    duration = time.time() - start_time
    qps = total * 1.0 / duration

    print(opt.len_class, metric_out([metric]), metric_out([metric_weighted]))
    print("Unreachable: {} {}, Ratio: {}, QPS: {}".format(unreachable, total, round(unreachable*1.0/total, 3), round(qps, 2)))
    print('[Info] CSSRNN Test ({}) Finished.'.format(opt.len_class))
