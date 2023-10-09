"""
Author: Omid Isfahani Alamdari
Date: 2023-10-02
Description: This scripts computes and store shortest path precomputations.
"""

import argparse
from datetime import datetime
import math
import time
import networkx as nx
import csv


def find_min_max_init_charge(cons_seq, batt_cap):
    cons_seq = [math.ceil(c) for c in cons_seq]
    init_charge = 0
    charge_so_far = init_charge
    
    max_sum_subarray = -batt_cap - 1
    max_sum_edning_here = 0

    prefix_sum = 0
    min_prefix_sum = batt_cap
    for i in range(0, len(cons_seq)):
        # for max_init_charge: find the minimum amount of prefix sum
        prefix_sum = prefix_sum + cons_seq[i]
        if prefix_sum < min_prefix_sum:
            min_prefix_sum = prefix_sum
        # find subarray with max sum ending at i
        max_sum_edning_here = max(max_sum_edning_here + cons_seq[i], cons_seq[i])
        if max_sum_subarray < max_sum_edning_here:
            max_sum_subarray = max_sum_edning_here
        
        # EV can not follow a part of the route (max subarray) with the sum of 
        # consecutive consumptions that is larger than the battery cap. Thus return -1.
        if max_sum_subarray > batt_cap:
            return -1, -1

        # 3 -6 2 -2 4 3

        # -3 5 -1 -4 1 ==> min charge = 2   cap = 7
        
        # charge at this stop
        charge = charge_so_far - cons_seq[i]
        if charge < 0:
            init_charge += -1 * charge
            
            # if at some point we need to have init_charge larger than the cap,
            # the path is not admissible.
            if init_charge > batt_cap:
                return -2, -2
            charge_so_far = 0
        elif charge > batt_cap:
            charge_so_far = batt_cap
        else:
            charge_so_far = charge
    max_init_charge = batt_cap + min_prefix_sum if min_prefix_sum <= 0 else batt_cap
    return init_charge, max_init_charge

def SP_precompute(G, charger_nodes, battery_cap, output_filename, reverse_network=False):
    output_file = open(output_filename, 'w')
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["src", "dst", "len", "dur", "min_init_charge", "max_init_charge", "cons"])

    for charger_id in charger_nodes:
        try:
            print(f"Computing shortest paths from {charger_id} at {datetime.now()}")
            durations, paths = nx.single_source_dijkstra(G, charger_id, weight='traveltime')
            
            for dest_id, path in paths.items():
                # sequence of consumptions of the path
                consumptions = []
                for sp_i in range(len(path)-1):
                    consumptions.append(float(G.edges()[path[sp_i], path[sp_i+1]]['consumption']))
                
                consumptions = consumptions[::-1]
                min_init_charge, max_init_charge = find_min_max_init_charge(consumptions, battery_cap)

                # spatial length of the path
                path_length = 0.0
                for sp_i in range(len(path)-1):
                    path_length += float(G.edges()[path[sp_i], path[sp_i+1]]['length'])
                
                formatted_values = [
                    "{:.3f}".format(value) if value != 0 else str(int(value))
                    for value in [path_length, durations[dest_id], min_init_charge, max_init_charge, sum(consumptions)]
                ]

                if reverse_network:
                    csv_writer.writerow([dest_id, charger_id, *formatted_values])
                else:
                    csv_writer.writerow([charger_id, dest_id, *formatted_values])

        except (nx.NodeNotFound) as e:
            print("There is no path to: %s" % charger_id)
    output_file.close()

def main():
    parser = argparse.ArgumentParser(description="Precomputeing shortes paths")
    parser.add_argument("input_road_network", type=str, help="Path to input road network GraphML file")
    parser.add_argument("nodes_from_chargers_output", type=str, default="reachable_nodes_from_chargers.csv", help="Output CSV file for reachable nodes (default: reachable_nodes_from_chargers.csv)")
    parser.add_argument("chargers_from_nodes_output", type=str, default="reachable_chargers_from_nodes.csv", help="Output CSV file for reachable chargers (default: reachable_chargers_from_nodes.csv)")
    parser.add_argument("--battery_cap", type=int, default=136800000, help="Battery capacity (Joule) (default: 136800000)")

    args = parser.parse_args()

    road_network = nx.read_graphml(args.input_road_network)
    nodes_from_chargers_output = args.nodes_from_chargers_output
    chargers_from_nodes_output = args.chargers_from_nodes_output
    battery_cap = args.battery_cap

    charger_nodes = set()
    for node, node_data in road_network.nodes(data=True):
        if 'charger_id' in node_data:
            charger_nodes.add(node)
    print(f"Computing shortest paths from {len(charger_nodes)} chargers ..." )

    st = time.time()
    SP_precompute(road_network, charger_nodes, battery_cap, nodes_from_chargers_output, False)
    et = time.time()
    print('Execution time for reachable_nodes_from_chargers :', et - st, 'seconds')
    
    road_network2 = road_network.reverse(copy=True)
    st = time.time()
    SP_precompute(road_network2, charger_nodes, battery_cap, chargers_from_nodes_output, True)
    et = time.time()
    print('Execution time for reachable_chargers_from_nodes :', et - st, 'seconds')


if __name__ == '__main__':
    main()