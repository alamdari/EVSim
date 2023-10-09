import argparse
from datetime import datetime
import math
import time
import networkx as nx
import csv
import pandas as pd
import numpy as np
import gzip
import json
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import BallTree

class BallTreeIndex:
    def __init__(self,lat_longs):
        self.lat_longs = np.radians(lat_longs)
        self.ball_tree_index = BallTree(self.lat_longs, metric='haversine')

    def query_radius(self, query, radius, r_d):
        RADIANT_TO_KM_CONSTANT = 6367
        radius_km = radius/1e3
        radius_radiant = radius_km / RADIANT_TO_KM_CONSTANT
        query = np.radians(np.array([query]))
        indices = self.ball_tree_index.query_radius(query,
                                                    r=radius_radiant,
                                                    return_distance=r_d)
        return indices
    
    def query(self,query):
        RADIANT_TO_KM_CONSTANT = 6367
        query = np.radians(np.array([query]))
        result = self.ball_tree_index.query(query, k=1, return_distance=True)
        return (result[0]*RADIANT_TO_KM_CONSTANT*1000.0, result[1])

def haversine_distance(x1, y1, x2, y2):
    R = 6371000 #meters
    lat_rad1 = radians(y1)
    lon_rad1 = radians(x1)
    lat_rad2 = radians(y2)
    lon_rad2 = radians(x2)
    return 2*R * asin(sqrt(sin((lat_rad2-lat_rad1)/2)**2 + cos(lat_rad1)*cos(lat_rad2)*(sin((lon_rad2-lon_rad1)/2)**2)))

def read_imn(filepath):
    user_imns = {}
    imn_filedata = gzip.GzipFile(filepath, 'r')
    for row in imn_filedata:
        if len(row) <= 1:
            print('new file started ;-)')
            continue
        
        user_obj = json.loads(row)
        uid = user_obj['uid']
        del user_obj['uid']
        if len(user_obj.keys()) != 0:
            user_imns[uid] = user_obj
    return user_imns

def verify_min_init_charge(cons_seq, init_charge, batt_cap):
    cons_seq = [math.ceil(c) for c in cons_seq]
    charge_so_far = init_charge
    #print(charge_so_far)
    for i in range(0, len(cons_seq)):
        charge = charge_so_far - cons_seq[i]
        if charge < 0:
            #print(charge, charge_so_far, cons_seq[i])
            return False, None
        elif charge > batt_cap:
            charge_so_far = batt_cap
        else:
            charge_so_far = charge
    return True, charge_so_far

def compute_sp_and_final_charge(src, dst, road_network, init_charge, battery_cap):
    """
    Returns the final remaining charge of the vehicle
    ----------
    src : string
        OSM id of the starting node
    dst : string
        OSM id of the destination node
    init_charge : numeric
        Amount of charge the vehicle has before this trip
    
    Returns
    -------
    remaining_charge : numeric
    """
    # do a SP to know the final charge
    try:
        length, path = nx.single_source_dijkstra(road_network, src, dst, weight='traveltime')
    except (nx.NodeNotFound, nx.NetworkXNoPath) as e:
        print("There is no path to: %s" % dst)
        return None, None, None, None, None
    
    # making the list of consumptions
    consumptions = []
    for sp_i in range(len(path)-1):
        consumptions.append(float(road_network.edges()[path[sp_i], path[sp_i+1]]['consumption']))

    duration = 0.0
    for sp_i in range(len(path)-1):
        duration += float(road_network.edges()[path[sp_i], path[sp_i+1]]['traveltime'])

    length_sp = 0.0
    for sp_i in range(len(path)-1):
        length_sp += float(road_network.edges()[path[sp_i], path[sp_i+1]]['length'])
    
    # verify if the path is feasible and what is the final charge
    is_feasible, remaining_charge = verify_min_init_charge(consumptions, init_charge, battery_cap)

    return is_feasible, remaining_charge, duration, length_sp, path


def main():
    home_charge = False  # does not work for the moment
    work_charge = False  # does not work for the moment

    initial_charge = 136800000
    battery_cap = 136800000
    battery_cap_kwh = 38
    home_work_power = 3.6 # kW

    # scenario = 1  # public stations + home charge
    # scenario = 2  # public stations + work charge
    scenario = 3  # public stations + home/work charge
    # scenario = 4  # only public stations and no home/work charge

    charge_amoount = battery_cap  # full recharge
    #charge_amoount = battery_cap / 2.0  # half recharge
    #charge_amoount = battery_cap / 4.0  # quarter recharge

    recharge_time = 3600*8  # full
    #recharge_time = 3600*4  # half
    #recharge_time = 3600*2  # quarter

    # k for: cost(station) = k* d(origin,station) + (1-k)*d(station,destination)
    # k = 0.5 selects the charger that leads to the least deviation
    # k = 0 selects the charger that is the closest to the destination
    # k = 1 selects the charger that is the closest to the origin
    k_first = 0.35
    k_second = 0.45
    k_charge = 0.2
    assert min(k_first, k_second, k_charge) >= 0, "k-weights cannot be negative"
    assert k_first + k_second + k_charge == 1, "k-weights should sum to 1"

    min_stay_time = 20 * 60

    # "... Walking from/to a different place to recharge is probably worth only if you plan 
    # to spend at least 1 hour or so." -- Mirco
    mod_discomfort_enabled = True
    min_stay_time_mod_discomfort = 60 * 60

    max_recharges = 3

    output_file_name = f"least_deviation_v4_scenario_{scenario}_{k_first}-{k_second}-{k_charge}_moderate_discomfort.csv"

    # if true, recharges at the closest charger
    # if false, recharges at the charger that has the least energy consumption.
    closest_charger = True

    #road_network = nx.read_graphml(args.input_road_network)
    road_network = nx.read_graphml('final_road_network.graphml')

    # Loading the precomputed information

    # for each node store the list of reachable stations (sorted by length)
        # reachable_charging_stations: dict of node -> [(sp_length, duration, min_init_charge, sp_consumption)]

    # for each station, store the dict of reachable destination nodes.
        # reachable_nodes_from_chargers: dict of station node -> {dest node -> (sp_length, duration, min_init_charge, sp_consumption)}

    # if the closest (by lenght of route) one does not work, 

    reachable_charging_stations = pd.read_csv('reachable_chargers_from_nodes.csv')
    reachable_charging_stations = reachable_charging_stations.set_index("src").sort_index()

    print(reachable_charging_stations.shape)
    #print(reachable_charging_stations.info())

    reachable_nodes_from_chargers = pd.read_csv('reachable_nodes_from_chargers.csv')
    reachable_nodes_from_chargers = reachable_nodes_from_chargers.set_index(["src", "dst"]).sort_index()
    
    print(reachable_nodes_from_chargers.shape)
    #print(reachable_nodes_from_chargers.info())

    

    # Building a BallTree index to perform the 1-NN search for the closest nodes to a given point
    network_nodes = []
    network_ids = []
    for n in road_network.nodes(data=True):
        network_nodes.append([float(n[1]['y']), float(n[1]['x'])])
        network_ids.append(n[0])
    print(len(network_nodes), len(network_ids))
    ball_tree_index = BallTreeIndex(np.array(network_nodes))

    # Reading the IMN file -- loading data of March - April only
    #imn_filepath = 'tuscany_five_prov_imn_sample_2017_0304_EV.json.gz'  # Orig Agnese
    imn_filepath = 'data/geolife_imns.json.gz'  # Omid's Embedding sample
    user_imns = read_imn(imn_filepath)
    
    # output file for the results
    output_file = open(output_file_name, 'w')
    csv_writer = csv.writer(output_file)

    row = ["uid", "movid", "sim_length", "mov_length", "sim_duration", "mov_duration", "start_ts", "end_ts", "consumption", "legs_consumptions", 
    "emergency", "scenario", "recharges", "start_osmid", "end_osmid", "start_charge", "stay_charge", "start_loc_type", "station_recharge_amounts"]
    csv_writer.writerow(row)
    ccccc = 0
    st = time.time()
    for uid, imn in user_imns.items():
        print("user ", uid)

        # reloading reachable_charger again to ensure previous user's data is not included
        # with open('reachable_charger.pickle', 'rb') as handle:
        #     reachable_charger = pickle.load(handle)

        # obtaining home and work location
        home_loc = (imn['location_prototype']['0'][1], imn['location_prototype']['0'][0])
        work_loc = (imn['location_prototype']['1'][1], imn['location_prototype']['1'][0])

        if home_loc == work_loc:
            print("Skipping User - home and work locations are the same")
            continue

        # adding home / work chargers based on the scenario

        # obtain home and work osmid
        query_res = ball_tree_index.query(query=home_loc)
        home_osmid = network_ids[query_res[1][0][0]] if query_res[1].size > 0 else -1
        home2node_dist = query_res[0][0][0]  # just for analysis of results

        query_res = ball_tree_index.query(query=work_loc)
        work_osmid = network_ids[query_res[1][0][0]] if query_res[1].size > 0 else -1
        work2node_dist = query_res[0][0][0]  # just for analysis of results

        print("Home: ", home_loc, home_osmid, home2node_dist)
        print("Work: ", work_loc, work_osmid, work2node_dist)

        # Replaces the above, enumerating all trips, not movements
        # Use traj id as temporal order
        sorted_od_pairs = sorted(imn['traj_location_from_to'].items(),
                                key=lambda item: int(item[0]))
        sorted_movement_ids = [str(imn['location_from_to_movement'][str(tuple(x[1]))])
                            for x in sorted_od_pairs]
        sorted_movements = [(k,imn['movement_prototype'][k]) for k in sorted_movement_ids]


        traj_list = list(imn['traj_location_from_to'].keys())

        print("!!!! ", len(traj_list), len(sorted_movements))
        iii = 0

        # the inital travel charge is set to the initial charge parameter
        travel_charge = initial_charge
        for idx in range(len(sorted_movements)):
            movid, mov = sorted_movements[idx]

            curr_traj_id = traj_list[idx]
            prev_traj_id = traj_list[idx-1] if idx>0 else -1

            print("curr_traj_id, prev_traj_id ", curr_traj_id, prev_traj_id)

            total_time = 0  # total time for travel and charging

            stay_charge_amount = 0

            trip_source_destination_sp_length = 0.0

            recharge_amounts = []

            # we keep this for the cases when things go wrong and we have to revert back to 
            # the inital charge before considering the next trip
            init_charge_before_start = travel_charge

            #travel_charge = travel_charge * 0.85

            start_point=(mov['object'][0][1], mov['object'][0][0])
            end_point=(mov['object'][-1][1], mov['object'][-1][0])

            time_diff = mov['object'][-1][2] - mov['object'][0][2]

            mov_length = mov['_length']
            mov_duration = mov['_duration']

            # finding the closest start and end osmids
            query_res = ball_tree_index.query(query=start_point)
            start_node_osmid = network_ids[query_res[1][0][0]] if query_res[1].size > 0 else -1
            start2node_dist = query_res[0][0][0]  # just for analysis of results

            query_res = ball_tree_index.query(query=end_point)
            end_node_osmid = network_ids[query_res[1][0][0]] if query_res[1].size > 0 else -1
            end2node_dist = query_res[0][0][0]  # just for analysis of results

            # I saw a couple of cases where the SP function freezes due to the fact that one of
            # the endpoints were extra charger nodes added to the road network.
            # If this is the case, we remove the dash and perform the SP on the original OSM node.
            # 12112414-full ==> 12112414
            dash_pos = start_node_osmid.find("-")
            if dash_pos != -1:
                start_node_osmid = start_node_osmid[:dash_pos]

            dash_pos = end_node_osmid.find("-")
            if dash_pos != -1:
                end_node_osmid = end_node_osmid[:dash_pos]

            print("new movement: ", uid, movid, start_node_osmid, end_node_osmid, travel_charge)
            if start_node_osmid == end_node_osmid:
                print("Start and end points are the same")
                continue

            curr_traj_start_ts = imn['tid_se_times'][str(curr_traj_id)][0]
            curr_traj_end_ts = imn['tid_se_times'][str(curr_traj_id)][1]

            #stay_time_at_loc = abs(mov['object'][0][2] - sorted_movements[idx-1][1]['object'][-1][2])
            stay_time_at_loc = imn['tid_se_times'][str(curr_traj_id)][0] - imn['tid_se_times'][str(prev_traj_id)][1] if idx>0 else 0
            print("stay_time_at_loc ", stay_time_at_loc, idx)

            start_location_type = "O"

            dist_to_home = haversine_distance(start_point[1], start_point[0], home_loc[1], home_loc[0])
            dist_to_work = haversine_distance(start_point[1], start_point[0], work_loc[1], work_loc[0])
            if (dist_to_home < 100 and scenario in (1,3)): # can use home charger
                if stay_time_at_loc > min_stay_time:
                    charge_amount_kwh = home_work_power * (stay_time_at_loc/3600.0)
                    charge_amount_138 = (charge_amount_kwh * battery_cap)/battery_cap_kwh
                    print("charge_amount_kwh, charge_amount_138: ", charge_amount_kwh, charge_amount_138)

                    travel_charge += charge_amount_138
                    # FIXME battery_cap is not in the same units as the charge obtained by home_work_power
                    travel_charge = battery_cap if travel_charge > battery_cap else travel_charge
                    stay_charge_amount = travel_charge - init_charge_before_start
                    start_location_type = "H"
                    print ("CHARGE AT HOME ", travel_charge)
            elif (dist_to_work < 100 and scenario in (2,3)): # can use home charger
                if stay_time_at_loc > min_stay_time:
                    charge_amount_kwh = home_work_power * (stay_time_at_loc/3600.0)
                    charge_amount_138 = (charge_amount_kwh * battery_cap)/battery_cap_kwh
                    print("charge_amount_kwh, charge_amount_138: ", charge_amount_kwh, charge_amount_138)

                    travel_charge += charge_amount_138
                    # FIXME battery_cap is not in the same units as the charge obtained by home_work_power
                    travel_charge = battery_cap if travel_charge > battery_cap else travel_charge
                    stay_charge_amount = travel_charge - init_charge_before_start
                    start_location_type = "W"
                    print ("CHARGE AT WORK ", travel_charge)
            elif mod_discomfort_enabled and stay_time_at_loc > min_stay_time_mod_discomfort:
                # When the stay point is not work / home
                # search for the stations around
                print("Moderate discomfort ... ", stay_time_at_loc)
                query_res = ball_tree_index.query_radius(start_point, 100, True)
                print(query_res)
                #home_osmid = network_ids[query_res[1][0][0]] if query_res[1].size > 0 else -1
                nearby_chargers = []               
                for i in range(len(query_res[0][0])):
                    node_id = network_ids[query_res[0][0][i]]
                    if 'speed' in road_network.nodes[node_id]: # this indicates that the node is charger
                        # charger id, distance, speed
                        nearby_chargers.append((node_id, 
                                                query_res[1][0][i], 
                                                road_network.nodes[node_id]['speed']))

                print(nearby_chargers)
                if len(nearby_chargers) > 0:
                    mod_discomfort_charger = sorted(nearby_chargers, key=lambda x: x[2], reverse=True)[0]
                    print(mod_discomfort_charger)
                    mod_discomfort_charger_speed = mod_discomfort_charger[2]

                    charge_amount_kwh = mod_discomfort_charger_speed * (stay_time_at_loc/3600.0)
                    charge_amount_138 = (charge_amount_kwh * battery_cap)/battery_cap_kwh
                    print("charge_amount_kwh, charge_amount_138: ", charge_amount_kwh, charge_amount_138)
                    travel_charge += charge_amount_138
                    travel_charge = battery_cap if travel_charge > battery_cap else travel_charge
                    stay_charge_amount = travel_charge - init_charge_before_start
                    start_location_type = "O"
                    print ("[MODERATE DISCOMFORT] CHARGE AT NON WORK/HOME STATION ", travel_charge)
                else:
                    print("No charger available within 100 meteres.. No recharge.")

            reached_destination = False
            no_charges = 0  # to avoid infinite loop for searching for chargers
            total_time = 0
            consumption = 0
            legs_consumptions = []
            path_nodes = []
            start_node = start_node_osmid
            while not reached_destination and no_charges <= max_recharges:
                no_closeby_charger = False  # this is for setting emergency type

                # do a direct SP
                is_feasible, remaining_charge, duration, length_sp, path = compute_sp_and_final_charge(start_node, end_node_osmid, road_network, travel_charge, battery_cap)
                if path is None:
                    print("No path!!", uid, movid)
                    break

                if no_charges == 0:
                    trip_source_destination_sp_length = length_sp
                print("direct SP ", is_feasible, remaining_charge)

                if is_feasible:
                    # Done with this source and destination, record this and continue to the next 
                    # reached destination
                    reached_destination = True
                    consumption += travel_charge - remaining_charge
                    legs_consumptions.append(consumption)
                    travel_charge = remaining_charge
                    print("Without charge reached destination ", travel_charge, consumption)
                    path_nodes.extend(path)
                    total_time += duration
                    break

                else:
                    ccccc += 1
                    print("Charging needed", movid, travel_charge, is_feasible, remaining_charge)

                    '''sp_info : dict
                            Dict that contains precomputed information. Keys include: 'length' that has 
                            the spatial length of the path, 'duration' that has the precomputed temporal 
                            distance of the path. If init_charge is between 'min_init_charge' and
                            'max_init_charge', the final charge will be init_charge - 'cons'.
                            'cons' specify the sum of consumptions of the path.'''
                    # if sp_info['min_init_charge'] < available_charge < sp_info['max_init_charge']:
                    #         remaining_charge = available_charge - sp_info['cons']
                    #     else:

                    travel_charge_before = travel_charge
                    src_node = int(start_node)
                    # Find all the reachable charging stations (rcs_list) from start_node, considering that 
                    # travel_charge  must be higher than min_init_charge. (we sort in non-decreasing order of the  
                    # length of the path, so that closest one is the first one)
                    #rcs_list = reachable_charging_stations[(reachable_charging_stations['src']==src_node)&(reachable_charging_stations['dst']!=int(src_node))&(reachable_charging_stations['min_init_charge']!=-1)&(reachable_charging_stations['min_init_charge']<travel_charge)].sort_values('len').values.tolist()
                    rcs_list = reachable_charging_stations.loc[[src_node]].query('dst != %s and min_init_charge != -1 and min_init_charge < %s' % (int(src_node), travel_charge)).sort_values('len').values.tolist()
                    # if there are no chargers available, this is an emergency
                    if len(rcs_list) == 0:
                        #print("EMERGENCY RCS")
                        # EMERGENCY condition: There are no charging stations in the vicinity
                        no_closeby_charger = True
                        break  # can not reach destination
                    #print(rcs_list[0])
                    print("Found reachable chargers ", len(rcs_list), rcs_list[0])
                    # only summing the length of two legs and putting every info from DFs in a dict
                    #"dst", "src", "len", "dur", "min_init_charge", “max_init_charge”, "cons"
                    reachable_chargers = list(map(lambda x: int(x[0]), rcs_list))

                    sp_precomp_info = []
                    for charger_info in rcs_list:
                        # charger_info[0] is the id of the charger
                        #charger2dst_info = reachable_nodes_from_chargers_filtered[(reachable_nodes_from_chargers_filtered['src']==int(charger_info[0]))].values.tolist()
                        try:
                            charger2dst_info = reachable_nodes_from_chargers.loc[[(int(charger_info[0]), int(end_node_osmid))]].values.tolist()
                        except (KeyError) as e:
                            continue
                        if len(charger2dst_info)>0:
                            charger2dst_info = charger2dst_info[0]
                        else:
                            continue

                        expected_recharge_time = 3600.0  # default value
                        travel_charge_leg1 = min(travel_charge, charger_info[4]) - charger_info[5]
    #                       
                        full_battery_recharge_time = 3600*battery_cap_kwh/road_network.nodes[str(int(charger_info[0]))]['charge_power']
                        expected_recharge_time = full_battery_recharge_time*(battery_cap-travel_charge_leg1)/battery_cap
                        # sp_precomp_info will contain the info of charger along with two legs
                        # charger, len1, len2, dur1, dur2, 
                        # min_init_charge1, min_init_charge2, max_init_charge1, max_init_charge2, 
                        # cons1, cons2, (added) exp_recharge_time
                        sp_precomp_info.append([
                            charger_info[0], charger_info[1], charger2dst_info[0], charger_info[2], charger2dst_info[1],
                            charger_info[3], charger2dst_info[2], charger_info[4], charger2dst_info[3],
                            charger_info[5], charger2dst_info[4], expected_recharge_time
                        ])

                    # Sort sp_precomp_info based on the logic of k
                    # cost(station) = k* d(origin,station) + (1-k)*d(station,destination)

                    sp_precomp_info = sorted(sp_precomp_info,
                                            key = lambda x:k_first*x[1] + k_second*x[2] + k_charge*x[11])

                    # the first charger in sp_precomp_info is the one to go for recharging
                    # then, the source will be that charger and we try to reach the destination
                    print(sp_precomp_info)
                    selected_station = str(sp_precomp_info[0][0])
                    first_leg_consumption = float(sp_precomp_info[0][9])
                    
                    # CHECK faster version:
                    selected_station = str(sp_precomp_info[0][0])
                    first_leg_consumption = float(sp_precomp_info[0][9])
                    first_leg_max_init_charge = float(sp_precomp_info[0][7])
                    print("first_leg_consumption ", first_leg_consumption, " first_leg_max_init_charge", first_leg_max_init_charge)
                    print("**", selected_station, "**")

                    if travel_charge < first_leg_max_init_charge:
                        travel_charge = travel_charge - first_leg_consumption
                    else:
                        travel_charge = first_leg_max_init_charge - first_leg_consumption

                    # print("Charge at ", selected_station, is_feasible, remaining_charge)

                    print("Reached to charger with charge: ", travel_charge)

                    # update travel charge for the next trip from charger to destination
                    # consumption += travel_charge - remaining_charge
                    # legs_consumptions.append(consumption)
                    # travel_charge = remaining_charge
                    travel_charge += charge_amoount
                    travel_charge = battery_cap if travel_charge > battery_cap else travel_charge

                    no_charges += 1

                    # path_nodes.extend(path[:-1] + [selected_station+"-"+recharge_type])
                    # total_time += duration + recharge_time

                    print("Left charger with charge: ", travel_charge)
                    #recharge_amounts.append([selected_station, remaining_charge, travel_charge])

                    # set the start_node_osmid to the charger and try to reach the destination
                    start_node = str(selected_station)

            # end while

            # end while 

            if not reached_destination:
                if no_closeby_charger and no_charges==0:
                    # EMERGENCY condition: There are no charging stations in the vicinity and the origin is the initial origin
                    # output the result with -1 for some fields
                    row = [uid, movid, trip_source_destination_sp_length, mov_length, -1, mov_duration, curr_traj_start_ts, curr_traj_end_ts, -1, json.dumps(legs_consumptions), 1, scenario, -1, start_node_osmid, end_node_osmid, init_charge_before_start, stay_charge_amount, start_location_type, json.dumps(recharge_amounts)]
                    csv_writer.writerow(row)
                    # setting the travel charge to initial charge to avoid consecutive emergencies!
                    travel_charge = initial_charge
                else:
                    # EMERGENCY condition: This happens when none of the chargers could help the agent to reach the destination
                    # output the result with -2 for some fields
                    print("EMERGENCY**", uid, movid)
                    charger_present = len([x for x in path_nodes if 
                                        ("full" in x or "half" in x or 
                                        "quarter" in x or "hour" in x or 
                                        "halfhour" in x)])
                    
                    row = [uid, movid, trip_source_destination_sp_length, mov_length, -2, mov_duration, curr_traj_start_ts, curr_traj_end_ts, -2, json.dumps(legs_consumptions), 1, scenario, charger_present, start_node_osmid, end_node_osmid, init_charge_before_start, stay_charge_amount, start_location_type, json.dumps(recharge_amounts)]
                    csv_writer.writerow(row)
                    # setting the travel charge to initial charge to avoid consecutive emergencies!
                    travel_charge = initial_charge
            else:
                # An in-between charger is found and we can use that to reach the destination
                # But we should identify the in-between charger that leads to the least amount of
                # consumption.

                #consumption, total_time, path_nodes = sorted(feasible_solutions, key=lambda x: x[0])[0]  # least dev.
                #consumption, total_time, path_nodes = feasible_solutions[0]  # closest 
                print("writing Output ", consumption, total_time, len(path_nodes))

                # output the result
                charger_present = len([x for x in path_nodes if 
                                        ("full" in x or "half" in x or 
                                        "quarter" in x or "hour" in x or 
                                        "halfhour" in x)])
                
                # sim_length = traj_length(node_positions)
                sim_length = 0.0
                
                row = [uid, movid, sim_length, mov_length, total_time, mov_duration, curr_traj_start_ts, curr_traj_end_ts, consumption, json.dumps(legs_consumptions), 0, 
                        scenario, charger_present, start_node_osmid, end_node_osmid, init_charge_before_start, stay_charge_amount, start_location_type, json.dumps(recharge_amounts)]
                csv_writer.writerow(row)


            #uid, movid, sim_length, mov_length, sim_duration, mov_duration, consumption, emergency?, scenario, recharges, sim_traj, sim_path

            iii += 1
        #break
        #remove_charge_edges(road_network, new_chargers_hw)
    et = time.time()
    print('Execution time:', et - st, 'seconds')

    output_file.close()

    print("FINISH ", datetime.now())

    #         # end while
    #         if not reached_destination:
    #             print("EMERGENCY**", uid, movid)
    #             travel_charge = initial_charge

                
    #         #uid, movid, sim_length, mov_length, sim_duration, mov_duration, consumption, emergency?, scenario, recharges, sim_traj, sim_path
            
    #         iii += 1
    #         et_trip = time.time()
    #         #print('Trip Execution time:', et_trip - st_trip, 'seconds')
    #         print("Trip execution time ", uid, movid, start_node_osmid, end_node_osmid, curr_traj_start_ts, curr_traj_end_ts, et_trip - st_trip)
    #     #break
        
    # et = time.time()
    # print('Execution time:', et - st, 'seconds')

    # #output_file.close()

    # print("FINISH ", datetime.now())

            

if __name__ == '__main__':
    main()