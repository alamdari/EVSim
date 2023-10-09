#import pandas as pd
import networkx as nx
import osmnx as ox
import argparse


def save_graphml(G, filepath=None, encoding="utf-8"):
    # Copied partially from OSMnx library.
    # https://github.com/gboeing/osmnx/
    # save_graphml function from io of OSMnx throwing errors while saving the DiGraph.
    # G.edges() does not identify key=True for DiGraph.

    # stringify all the graph attribute values
    for attr, value in G.graph.items():
        G.graph[attr] = str(value)

    # stringify all the node attribute values
    for _, data in G.nodes(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    # stringify all the edge attribute values
    for _, _, data in G.edges(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    nx.write_graphml(G, path=filepath, encoding=encoding)

def main():
    parser = argparse.ArgumentParser(description="Generate a road network graph within a specified bounding box.")
    parser.add_argument("output_road_network", type=str, help="Path to output road network GraphML file")
    # Arguments for bounding box coordinates
    parser.add_argument('--min_lat', type=float, required=True, help='Minimum latitude')
    parser.add_argument('--min_lon', type=float, required=True, help='Minimum longitude')
    parser.add_argument('--max_lat', type=float, required=True, help='Maximum latitude')
    parser.add_argument('--max_lon', type=float, required=True, help='Maximum longitude')

    args = parser.parse_args()

    study_min_lat, study_min_lon = args.min_lat, args.min_lon
    study_max_lat, study_max_lon = args.max_lat, args.max_lon

    # study_min_lat, study_min_lon = args.min_lat+0.5, args.min_lon+0.5
    # study_max_lat, study_max_lon = args.max_lat-0.5, args.max_lon-0.5

    output_road_network = args.output_road_network

    # Download the road network from the specified bounding box
    road_network = ox.graph_from_bbox(study_max_lat, study_min_lat, 
                        study_max_lon, study_min_lon, 
                        network_type="drive")

    #print(type(road_network))
    # Convert road network to simple DiGraph to avoid parallel edges
    road_network_digraph = ox.utils_graph.get_digraph(road_network, 'length')

    #print(type(road_network_digraph))

    save_graphml(road_network_digraph, output_road_network)

    
    #nx.write_graphml(road_network_digraph, output_road_network, encoding="utf-8")

if __name__ == '__main__':
    main()
