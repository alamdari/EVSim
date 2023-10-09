"""
Author: Omid Isfahani Alamdari
Date: 2023-10-02
Description: This scripts adds important attributes to the road network.
"""

import argparse
import json
import networkx as nx
import xarray as xr
import osmnx as ox
import numpy as np
import math


# PARAMETERS OF THE VEHICLE TO SIMULATE
# Vehicle frontal area (m^2)
FRONTAL_AREA = 2.27
# Vehicle mass (kg)
VEHICLE_MASS = 1580
# Driver mass (kg)
DRIVER_MASS = 90
# Coefficient of aerodynamic resistance
AERODYNAMIC_RESISTANCE = 0.29
# Coefficient of rolling resistance
ROLLING_RESISTANCE = 0.012
# Battery capacity (kWh)
BATTERY_CAPACITY = 38
# Total mass (kg)
TOTAL_MASS = VEHICLE_MASS + DRIVER_MASS
# Transmission efficiency
TRANSMISSION_EFFICIENCY = 0.97
# Electric motor efficiency
MOTOR_EFFICIENCY = 0.95
# Auxiliary power (W)
AUXILIARY_POWER = 0
# Battery charging efficiency
BATTERY_CHARGE_EFFICIENCY = 0.95
# Battery discharge efficiency
BATTERY_DISCHARGE_EFFICIENCY = 0.98
# Gravitational acceleration (m/s^2)
GRAVITY_ACCELERATION = 9.81
# Air density (kg/m^3)
AIR_DENSITY = 1.2041
# Regeneration ratio (G = 0.35 in ECO MODE)
REGENERATION_RATIO = 0.35

def calculate_battery_consumption(distance, time_gap, speed, acceleration, slope_angle):
    # Rolling resistance
    rolling_resistance_force = ROLLING_RESISTANCE * TOTAL_MASS * GRAVITY_ACCELERATION * math.cos(slope_angle)
    # Aerodynamic resistance
    aerodynamic_resistance_force = 0.5 * FRONTAL_AREA * AERODYNAMIC_RESISTANCE * AIR_DENSITY * speed**2
    # Gravitational force
    gravitational_force = TOTAL_MASS * GRAVITY_ACCELERATION * math.sin(slope_angle)
    # Inertial force
    inertial_force = 1.05 * TOTAL_MASS * acceleration
    # Total force
    total_force = rolling_resistance_force + aerodynamic_resistance_force + gravitational_force + inertial_force
    # Mechanical traction power
    total_power = total_force * speed
    
    motor_output_power = 0
    if total_power >= 0:
        motor_output_power = total_power / TRANSMISSION_EFFICIENCY
    else:
        motor_output_power = REGENERATION_RATIO * total_power * TRANSMISSION_EFFICIENCY
    
    motor_input_power = 0
    if motor_output_power >= 0:
        motor_input_power = motor_output_power / MOTOR_EFFICIENCY
    else:
        motor_input_power = motor_output_power * MOTOR_EFFICIENCY
    
    battery_power = motor_input_power + AUXILIARY_POWER
    
    # Battery modeling
    battery_energy = 0
    if battery_power >= 0:
        battery_energy = battery_power * time_gap / BATTERY_DISCHARGE_EFFICIENCY
    else:
        battery_energy = battery_power * time_gap * BATTERY_DISCHARGE_EFFICIENCY
    
    # Calculate Delta State of Charge (SoC)
    KWH_TO_JOULES = 3600 * 1e3
    delta_soc = battery_energy / (BATTERY_CAPACITY * KWH_TO_JOULES)
    
    return delta_soc

def correct_edge_max_speed(G, default_max_speed):
    edge_attributes = {}
    for source, target, edge_data in G.edges(data=True):
        if 'maxspeed' in edge_data:
            if isinstance(edge_data['maxspeed'], str):
                try:
                    speed_val = int(edge_data['maxspeed'])
                except ValueError:
                    speed_val = default_max_speed
                edge_attributes[(source, target)] = speed_val
                
            elif isinstance(edge_data['maxspeed'], list):
                speed_vals = [s for s in edge_data['maxspeed'] if s.isdigit()]
                try:
                    speed_val = int(max(speed_vals))
                except ValueError:
                    speed_val = default_max_speed
                edge_attributes[(source, target)] = speed_val
        else:
            edge_attributes[(source, target)] = default_max_speed
    nx.set_edge_attributes(G, edge_attributes, 'maxspeed')

def assign_node_elevations(G, geotiff_path, min_lon, min_lat, max_lon, max_lat):
    # loading the elevation data using rasterio
    data_elevation = xr.open_dataset(geotiff_path, engine="rasterio").squeeze().drop("band")
    
    # slicing only the desired part (tiles may expand the bbox area)
    data_array = data_elevation.sel(x=slice(min_lon,max_lon),y=slice(max_lat,min_lat))['band_data']

    nodes_attributes = {}
    for i, (node, node_data) in enumerate(G.nodes(data=True)):
        node_lat = float(node_data['y'])
        node_lon = float(node_data['x'])
        elev_value = data_array.sel(x=node_lon, y=node_lat, method="nearest").values.item()
        nodes_attributes[node] = float(elev_value)

        if i % 10000 == 0:
            print(f"No. of elevations computed: {i}")
    
    nx.set_node_attributes(G, nodes_attributes, 'elevation')

def add_edge_slope(G, default_slope=0):
    edge_attributes = {}
    edge_attributes_slope_angle = {}
    for source, target, edge_data in G.edges(data=True):
        slope = float((G.nodes[source]['elevation']) - (G.nodes[target]['elevation']))
        if abs(slope) > float(edge_data['length']):
            slope = default_slope

        length = float(edge_data['length'])

        if slope != 0:
            hypotenuse = math.hypot(length, abs(slope))
            slope_angle = math.atan(slope / hypotenuse)
        else:
            slope_angle = 0
        
        edge_attributes[(source, target)] = slope
        edge_attributes_slope_angle[(source, target)] = slope_angle

    nx.set_edge_attributes(G, edge_attributes, 'slope')
    nx.set_edge_attributes(G, edge_attributes_slope_angle, 'slope_angle')

def add_traveltime_consumption(G):
    edge_attributes_traveltime = {}
    edge_attributes_consumption = {}
    for source, target, edge_data in G.edges(data=True):
        slope = float(edge_data['slope'])
        slope_angle = float(edge_data['slope_angle'])
        length = float(edge_data['length'])
        maxspeed = float(edge_data['maxspeed'])/3.6

        travel_time = length / (maxspeed/3.6)
        consumption = calculate_battery_consumption(length, travel_time, maxspeed, 0, slope_angle)
        consumption = float(consumption * (BATTERY_CAPACITY * (3600 * 1e3)))
        
        edge_attributes_traveltime[(source, target)] = travel_time
        edge_attributes_consumption[(source, target)] = consumption

    nx.set_edge_attributes(G, edge_attributes_traveltime, 'traveltime')
    nx.set_edge_attributes(G, edge_attributes_consumption, 'consumption')

def add_charging_stations(G, stations_dict, distance_threshold):
    keys, latitudes, longitudes, ch_powers = zip(*[(key, *values) 
                                                    for key, values in stations_dict.items()])
    nearest_nodes, dists = ox.nearest_nodes(G, longitudes, latitudes, return_dist=True)

    charge_power_attr = {node: stations_dict[node_id][2] 
                            for node, node_id, distance in zip(nearest_nodes, keys, dists) 
                                if distance <= distance_threshold}
    charger_id_attr = {node: node_id for node, node_id in zip(charge_power_attr, keys)}

    nx.set_node_attributes(G, charger_id_attr, name="charger_id")
    nx.set_node_attributes(G, charge_power_attr, name="charge_power")

def convert_node_dtypes(G):
    node_dtypes = {
        "elevation": float,
        "x": float,
        "y": float
    }
    for attr_name, dtype in node_dtypes.items():
        attrs = nx.get_node_attributes(G, attr_name)
        attrs_converted = {node: dtype(value) for node, value in attrs.items()}
        nx.set_node_attributes(G, attrs_converted, name=attr_name)

def convert_edge_dtypes(G):
    edge_dtypes = {
        "length": float,
        "traveltime": float,
        "consumption": float,
    }
    for attr_name, dtype in edge_dtypes.items():
        attrs = nx.get_edge_attributes(G, attr_name)
        attrs_converted = {edge: dtype(value) for edge, value in attrs.items()}
        nx.set_edge_attributes(G, attrs_converted, name=attr_name)


def main():
    parser = argparse.ArgumentParser(description="Preprocess road network data")
    parser.add_argument("input_road_network", type=str, help="Path to input road network GraphML file")
    parser.add_argument("elevation_data_file", type=str, help="Path to elevation data file")
    parser.add_argument("charging_stations_file", type=str, help="Path to charging stations JSON file")
    
    parser.add_argument("output_road_network", type=str, help="Path to output road network GraphML file")
    
    parser.add_argument("--min_lon", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--min_lat", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--max_lon", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--max_lat", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--default_max_speed", type=float, default=50, help="Default maximum speed")


    args = parser.parse_args()

    road_network = nx.read_graphml(args.input_road_network)
    elevation_data_file = args.elevation_data_file
    charging_stations = json.load(open(args.charging_stations_file,"r"))

    output_road_network = args.output_road_network
    
    default_max_speed = args.default_max_speed
    min_lat, min_lon = args.min_lat, args.min_lon
    max_lat, max_lon = args.max_lat, args.max_lon

    print(f"RN loaded :: Nodes#: {len(road_network.nodes())}, Edges#: {len(road_network.edges())}")

    convert_node_dtypes(road_network)  # to avoid string x and y attributes and convert to float
    
    assign_node_elevations(road_network, elevation_data_file, min_lon, min_lat, max_lon, max_lat)

    correct_edge_max_speed(road_network, default_max_speed)

    add_edge_slope(road_network, 0)
    
    add_traveltime_consumption(road_network)

    # The max distance threshold for assigning a charging stations to a node in the road network.
    max_assignment_dist_thresh = 500  # We suggest a threshold of 500m
    add_charging_stations(road_network, charging_stations, max_assignment_dist_thresh)

    convert_edge_dtypes(road_network)  # to avoid string edge attributes
    
    nx.write_graphml(road_network, output_road_network, encoding="utf-8")


if __name__ == '__main__':
    main()