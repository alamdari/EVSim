"""
Author: Omid Isfahani Alamdari
Date: 2023-10-02
Description: This scripts download the elevation data as GeoTIFF for a desired bbox.
"""

import argparse
import xarray as xr
import requests
import numpy as np
import rasterio
from rasterio.merge import merge
import os

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Save merged GeoTIFF data")
    parser.add_argument("output_file", required=True, help="Output GeoTIFF file path")
    parser.add_argument("--min_lon", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--min_lat", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--max_lon", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--max_lat", type=float, required=True, help="Maximum latitude")

    args = parser.parse_args()

    min_lat, min_lon = args.min_lat, args.min_lon
    max_lat, max_lon = args.max_lat, args.max_lon
    output_file = args.output_file

    params = {
        "bbox": ",".join(map(str, (min_lon, min_lat, max_lon, max_lat))),
        "collections": ["cop-dem-glo-30"],
    }

    stac_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-30"

    try:
        response = requests.get(stac_api_url + "/items", params=params)
        response.raise_for_status()

        # Get the list of items
        items = response.json()["features"]

        # Create a list to hold file paths of saved tiles
        tile_file_paths = []

        # Iterate through items and save each tile as a separate GeoTIFF file
        for i, item in enumerate(items):
            asset_url = item["assets"]["data"]["href"]
            print(f"The asset URL: {asset_url}")
            geo_tiff_response = requests.get(asset_url)
            geo_tiff_response.raise_for_status()
            
            print(type(geo_tiff_response.content))

            # Save the tile as a separate GeoTIFF
            tile_filename = f"tile_{i}.tif"
            with open(tile_filename, "wb") as tile_file:
                tile_file.write(geo_tiff_response.content)
            
            tile_file_paths.append(tile_filename)

        datasets = [rasterio.open(tile_file) for tile_file in tile_file_paths]

        # Merge all datasets into one
        merged_dataset, out_trans = merge(datasets)

        merged_profile = datasets[0].profile.copy()
        merged_profile.update({
                                "driver": "GTiff",
                                "height": merged_dataset.shape[1],
                                "width": merged_dataset.shape[2],
                                "transform": out_trans
                                })

        with rasterio.open(output_file, "w", **merged_profile) as dst:
            dst.write(merged_dataset)

        # Remove temp files after merging
        for file_name in tile_file_paths:
            file_path = os.path.join(os.path.dirname(__file__), file_name)
            os.remove(file_path)
        
        data_elevation = xr.open_dataset(output_file, engine="rasterio").squeeze().drop("band")

        data_array = data_elevation.sel(x=slice(min_lon,max_lon),y=slice(max_lat,min_lat))['band_data']

        data_array.plot.imshow(size=8,cmap=plt.cm.terrain,vmin=0.0,vmax=np.max(data_array))
        plt.gca().set_aspect('equal')
        plt.title('Terrain Elevation (meters)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(output_file.replace(".tif", ".png"), dpi=300, bbox_inches='tight')

    except Exception as e:
        print(f"Error fetching data: {e}")


if __name__ == '__main__':
    main()