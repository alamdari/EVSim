import pandas as pd
import numpy as np
import json
import gzip
import datetime
import networkx as nx
from networkx.readwrite import json_graph

import sys
sys.path.append('libs')

from trajectory import Trajectory
import trajectory_segmenter as trajectory_segmenter
from individual_mobility_network import build_imn

def key2str(k):
    if isinstance(k, tuple):
        return str(k)
    elif isinstance(k, datetime.time):
        return str(k)
    elif isinstance(k, np.int64):
        return str(k)
    elif isinstance(k, np.float64):
        return str(k)
    return k


def clear_tuples4json(o):
    if isinstance(o, dict):
        return {key2str(k): clear_tuples4json(o[k]) for k in o}
    return o


def agenda_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    elif isinstance(o, datetime.timedelta):
        return o.__str__()
    elif isinstance(o, Trajectory):
        return o.to_json()
    elif isinstance(o, nx.DiGraph):
        return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
    else:
        return o.__str__()

def main():
    # Segmentation parameters
    temporal_thr = 1200
    spatial_thr = 50
    max_speed = 0.07

    # min number of trajectories after segmentation for a user to be considered as
    # a user with enough mobility
    min_trajs_no = 10 

    points_df = pd.read_csv('data/filtered_points_w_header.csv')

    output_filename = 'data/geolife_imns.json.gz'

    for uid, user_data in points_df.groupby('id'):
        points = user_data.sort_values(by=['timestamp'])[
            ['longitude','latitude','timestamp']].values.tolist()
        trajs_dict = trajectory_segmenter.segment_trajectories(
            points, int(uid), temporal_thr=temporal_thr, spatial_thr=spatial_thr, 
            max_speed=max_speed)

        imh = {'uid': int(uid), 'trajectories': trajs_dict}

        if len(trajs_dict) <= min_trajs_no:
            continue

        imn = build_imn(imh)
        if imn is not None:
            imn["uid"] = uid
            json_str = '%s\n' % json.dumps(clear_tuples4json(imn), default=agenda_converter)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(output_filename, 'a') as fout:
                fout.write(json_bytes)

            print(f"IMN created for {uid} with {len(trajs_dict)} trajectories")

if __name__ == '__main__':
    main()
