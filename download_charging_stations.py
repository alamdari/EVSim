import argparse
import requests
import json


def main():
    default_powerKW = 22.0
    # a location in Beijing (for Geolife dataset)
    lat = 39.897435
    lon = 116.405998

    # This code is for representation only. You need to set the maxresults carefully, or 
    # set a proper sleep to avoid reaching API quota.
    
    # ATTENTION: The key must be set with your own API KEY frm the openchargemap portal. 
    parameters = {
        "output": "json",
        "latitude": lat,  
        "longitude": lon,
        "distance": 50,
        "maxresults": 100,
        "compact": True,
        "verbose": False,
        "key": "YOUR OCM API KEY"
    }

    ocm_results = {}

    response = requests.get("https://api.openchargemap.io/v3/poi/", params=parameters)
    
    # For simplicity, we take the max powerKW from the available connections.
    for oc in response.json():
        if 'Connections' in oc:
            max_power_kw = max(connection.get('PowerKW', default_powerKW) for connection in oc['Connections'])
        ocm_results[oc['ID']] = (oc['AddressInfo']['Latitude'], oc['AddressInfo']['Longitude'], max_power_kw)
    
    # save retrieved EV charging stations
    with open('ev_charging_stations.json', 'w') as fp:
        json.dump(ocm_results, fp)

if __name__ == '__main__':
    main()
