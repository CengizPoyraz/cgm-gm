import pandas as pd

station_data = pd.read_csv('station_list.csv')
station_codes = station_data.iloc[:,1]


def parse_phase_picks(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = []

    for line in lines:
        if line.startswith("EVENT"):
            event_id = line.split()[1]
        if line[:7] == '2023/02':  
            parts = line.split()
            depth = float(parts[4])
            magnitude = float(parts[9])
            src_lat = float(parts[2])
            src_lon = float(parts[3])
        elif len(line.split()) != 0:  
            if line.split()[0] in station_codes.values:
                parts = line.split()
                if parts[4] == 'Pg':
                    station_information = station_data[station_codes==parts[0]]
                    data.append({
                        'station': parts[0],
                        'time': parts[5] + ' ' + parts[6],
                        'eid': event_id + '_' + parts[0],
                        'src_lat': src_lat,
                        'src_lon': src_lon,
                        'sta_lat': float(station_information['Latitude']),
                        'sta_lon': float(station_information['Longitude']),
                        'depth': depth,
                        'mag': magnitude,
                        'Sampling Frequency': 100,
                        'Number of Time Steps': 6000,
                        'Delta t': 0,
                        'Min Frequency': 2,
                        'Max Frequency': 15,
                        'Number of Frequency Steps': 0,
                        'rup': float(parts[1]),
                    })
        
    return data


parsed_data = parse_phase_picks("phase_picks_6_7.txt")
df = pd.DataFrame(parsed_data)
df.to_csv("parsed_phase_picks.csv", index=False)
