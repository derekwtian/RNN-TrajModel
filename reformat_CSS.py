import argparse
import csv
import os
import pickle
from ast import literal_eval
import pandas as pd
import networkx as nx
import geopandas as gpd
import numpy as np
from tqdm import tqdm


def load_edges(edges_shp):
    edges = gpd.read_file(edges_shp)
    eid = edges['fid'].tolist()
    u = edges['u'].tolist()
    v = edges['v'].tolist()
    length = edges['length'].tolist()
    data = []
    for i in range(len(eid)):
        data.append([eid[i], u[i], v[i], length[i]])
    df = pd.DataFrame(data, columns=['eid', 'source', 'target', 'length'])
    print("Number of Segments: {}, {}".format(df.shape[0], edges.shape[0]))
    return df


def build_graph(road_file):
    G = nx.DiGraph(nodetype=int)
    edges = load_edges(road_file)
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        u = int(tmp['eid'])
        tar = int(tmp['target'])
        out_edges = edges.query('source == ' + str(tar))
        v_set = out_edges['eid'].tolist()
        out_length = out_edges['length'].tolist()
        for k in range(len(v_set)):
            G.add_edge(u, v_set[k], length=round(out_length[k], 3))
        # add length attribute for node (segment)
        lens = round(float(tmp['length']), 3)
        if G.has_node(u):
            G.nodes[u]['length'] = lens
        else:
            G.add_node(u, length=lens)
    return G


def gen_map(map_dir, out_dir):
    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    index = [i for i in range(nodes.shape[0])]
    nodes["fid"] = np.array(index, dtype=int)
    data = []
    nid_dict = {}
    for i in range(nodes.shape[0]):
        tmp = nodes.iloc[i]
        osmid = int(tmp['osmid'])
        fid = int(tmp['fid'])
        x = float(tmp['x'])
        y = float(tmp['y'])

        nid_dict[osmid] = fid
        data.append([fid, y, x])
    with open(os.path.join(out_dir, "nodeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)

    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    data = []
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        eid = int(tmp['fid'])
        u = int(tmp['u'])
        v = int(tmp['v'])
        points = tmp['geometry'].coords

        row = [eid, nid_dict[u], nid_dict[v]]
        row.append(len(points))
        for lon, lat in points:
            row += [float(lat), float(lon)]
        data.append(row)
    with open(os.path.join(out_dir, "edgeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)

    # G = build_graph(os.path.join(map_dir, "edges.shp"))
    # pickle.dump(G, open(os.path.join(out_dir, "road_graph"), "wb"))


def CSS_format(trajfile):
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    paths = []
    for i in tqdm(range(trajs.shape[0]), desc='data loading'):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        paths.append([item[0] for item in traj])
    print("Paths Number: {}".format(len(paths)))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--map_dir', type=str, default="../data/")
    parser.add_argument('--out_dir', type=str, default="../data/len_test")
    args = parser.parse_args()
    print(args)

    data_path = os.path.join(args.out_dir, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    map_path = os.path.join(args.out_dir, "map")
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    model_path = os.path.join(args.out_dir, "ckpt")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train = CSS_format(os.path.join(args.data_dir, "traj_train_100"))
    with open(os.path.join(data_path, "train_csstraj.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=',')
        fields_output_file.writerows(train)

    valid = CSS_format(os.path.join(args.data_dir, "traj_valid"))
    with open(os.path.join(data_path, "valid_csstraj.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=',')
        fields_output_file.writerows(valid)

    test = CSS_format(os.path.join(args.data_dir, "traj_test"))
    with open(os.path.join(data_path, "test_csstraj.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=',')
        fields_output_file.writerows(test)
    print(len(train), len(valid), len(test))

    gen_map(args.map_dir, map_path)
    print("Done")
