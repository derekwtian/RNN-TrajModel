import argparse
import os
import pickle
import time
from collections import OrderedDict

from main import read_data2, Config, MapInfo
from geo import Map, GeoPoint
import tensorflow as tf
from trajmodel import TrajModel
import pandas as pd


class RoutePlanner(object):
    def __init__(self, ckpt_dir):
        config = pickle.load(open(os.path.join(ckpt_dir, "support.pkl"), "rb"))
        config.printf()

        GeoPoint.AREA_LAT = 41.15
        roadnet = Map()
        roadnet.open(config.map_path)
        mapInfo = MapInfo(roadnet, config)

        if config.constrained_softmax_strategy == 'adjmat_adjmask':
            self.out = 'adjmat'
        else:
            self.out = 'sparse'

        model_scope = "Model"
        with tf.name_scope("Test"):
            with tf.variable_scope(model_scope, reuse=None):
                self.model = TrajModel(False, config, None, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = tf.InteractiveSession(config=sess_config)

        ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        print('try loading ' + ckpt_path)
        self.model.saver.restore(self.session, ckpt_path)
        print("restoring model trainable params from %s successfully" % ckpt_path)

    def batch_planning(self, true_paths, MAX_ITERS=300):
        gens = [[t[0]] for t in true_paths]
        pending = OrderedDict({i:None for i in range(len(true_paths))})

        while True:
            if len(pending) == 0:
                break
            inputs = [gens[i] + [true_paths[i][-1]] for i in pending]
            if len(inputs[0]) >= MAX_ITERS:
                break
            self.model.config.batch_size = len(inputs)
            vals = self.model.predict(self.session, inputs)

            chosen = vals[self.out][:, -1].reshape(-1).tolist()
            pending_trip_ids = list(pending.keys())
            for identity, choice in zip(pending_trip_ids, chosen):
                dest = true_paths[identity][-1]
                last = gens[identity][-1]
                adjList_ids = self.model.map.edges[last].adjList_ids
                if dest in adjList_ids:
                    gens[identity].append(dest)
                    del pending[identity]
                    continue
                if choice == -1:
                    del pending[identity]
                    continue
                gens[identity].append(choice)
                if choice == dest:
                    del pending[identity]
        return gens, true_paths

    def planning(self, o, d, max_seq_len=75):
        route = [o]
        step = 2
        while len(route) < max_seq_len and route[-1] != d:
            out_segs = list(self.G.out_edges(route[-1]))
            if (route[-1], d) in out_segs:
                route.append(d)
                break

            data = [route + [d]]
            vals = self.model.predict(self.session, data)
            route.append(vals['output'][-1])
            # step += 1
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
    parser.add_argument('--model', type=str, default="/home/tianwei/dataset/data_CSS/sanfran/ckpt")
    parser.add_argument('-batch_size', type=int, default=4096)
    parser.add_argument('-max_len', type=int, default=300)
    parser.add_argument("-save_preds", action="store_true", default=False)
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    planner = RoutePlanner(opt.model)

    config = planner.model.config
    routes, train, valid, test = read_data2(config.train_path, config.valid_path, config.test_path, config.data_size, config.max_seq_len)
    print("successfully read %d routes" % sum([len(train), len(valid), len(test)]))
    print("train:%d, valid:%d, test:%d" % (len(train), len(valid), len(test)))
    planner.model.config.data_size = len(routes)
    planner.model.eval(planner.session, valid, True, False)

    path_num = len(test)
    print("==> path_num: {}, batch_size: {}".format(path_num, opt.batch_size))
    start_time = time.time()
    preds = []
    truths = []
    for i in range(0, path_num, opt.batch_size):
        bat_data = test[i: i + opt.batch_size]
        gens, paths = planner.batch_planning(bat_data, MAX_ITERS=opt.max_len)
        preds.extend(gens)
        truths.extend(paths)
    duration = time.time() - start_time
    qps = path_num * 1.0 / duration
    print("Number of trajs predicted per second (QPS): {}".format(round(qps, 2)))
    print("truths: {}, preds: {}".format(len(truths), len(preds)))

    if opt.save_preds:
        pickle.dump(list(zip(truths, preds)), open(os.path.join(opt.model, "truths_gens.pkl"), "wb"))
    print("[Info] CSSRNN Inference Finished.")

    # # route = planner.planning(5772, 6046)
    # # print(route)
    #
    # test = pickle.load(open(os.path.join(opt.workspace, opt.test_data), "rb"))
    # test = test[opt.len_class]
    # max_step = min(config.max_seq_len - 1, opt.max_len)
    # debug_mode = False
    # scale_factor = 1.2
    #
    # metric = [0, 0, 0, 0]
    # metric_weighted = [0, 0, 0, 0]
    # unreachable = 0
    # total = 0
    #
    # pred_rows = []
    #
    # start_time = time.time()
    # for path in test:
    #     if len(path) < 3:
    #         continue
    #     # _, tmp = nx.single_source_dijkstra(planner.G, path[0], path[-1], weight='length')
    #     # max_step = math.ceil(scale_factor * len(tmp))
    #     pred = planner.planning(path[0], path[-1], max_step)
    #     total += 1
    #     if pred[-1] != path[-1]:
    #         unreachable += 1
    #     res, res_weighted = f1_metric(set(pred), set(path), planner.G.nodes)
    #
    #     if opt.save_pred:
    #         pred_rows.append([path[0], path[-1], pred[-1],
    #                           len(pred), pred,
    #                           len(path), path,
    #                           metric_out([res]),
    #                           metric_out([res_weighted])])
    #
    #     if debug_mode:
    #         print("=========>{}, {}".format(metric_out([res]), metric_out([res_weighted])))
    #         print("OD", path[0], path[-1])
    #         print("real", len(path), path)
    #         print("pred", len(pred), pred)
    #         # print("shortest", len(tmp), tmp == path, tmp)
    #
    #     metric[0] += res[0]
    #     metric[1] += res[1]
    #     metric[2] += res[2]
    #     metric[3] += res[3]
    #     metric_weighted[0] += res_weighted[0]
    #     metric_weighted[1] += res_weighted[1]
    #     metric_weighted[2] += res_weighted[2]
    #     metric_weighted[3] += res_weighted[3]
    #
    # duration = time.time() - start_time
    # qps = total * 1.0 / duration
    #
    # print(opt.len_class, metric_out([metric]), metric_out([metric_weighted]))
    # print("Unreachable: {} {}, Ratio: {}, QPS: {}".format(unreachable, total, round(unreachable*1.0/total, 3), round(qps, 2)))
    # print('[Info] CSSRNN Test ({}) Finished.'.format(opt.len_class))
    #
    # if opt.save_pred:
    #     with open(os.path.join(opt.workspace, "predicted_routes_{}-{}.txt".format(opt.len_class, max_step)), 'w') as fp:
    #         fields_output_file = csv.writer(fp, delimiter=',')
    #         fields_output_file.writerows(pred_rows)
    #     print(len(pred_rows), total)
