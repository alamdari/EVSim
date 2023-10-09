import pandas as pd
import os

def main():
    root_folder = 'data/Geolife Trajectories 1.3/Data/'
    # Read only the car and taxi trajectories that fall in the following time range
    study_modes = ['car', 'taxi']
    study_start_time = pd.Timestamp('2008-05-01')
    study_end_time = pd.Timestamp('2008-10-31')
    study_min_lat, study_min_lon, study_max_lat, study_max_lon = 39.5, 115.8, 40.8, 117.4

    all_points = []

    overlapping_trajectories = []
    folder_list = [os.path.join(root_folder, folder_name) 
                for folder_name in os.listdir(root_folder) 
                    if os.path.isdir(os.path.join(root_folder, folder_name))]

    for user_dir in folder_list:
        user_id = user_dir.split('/')[-1]
        labels_file_path = os.path.join(user_dir, 'labels.txt')
        
        try:
            labels_df = pd.read_csv(labels_file_path, delimiter='\t', 
                                    parse_dates = ['Start Time', "End Time"])
        except (FileNotFoundError):
            continue
        
        print(user_id)
        labels_df = labels_df[labels_df['Transportation Mode'].isin(study_modes)]

        traj_folder = os.path.join(user_dir, 'Trajectory')
        traj_files = [f for f in os.listdir(traj_folder) if f.endswith('.plt')]
        
        traj_filenames = []
        traj_start_times = []
        traj_end_times = []
        for traj_file in traj_files:
            traj_file_path = os.path.join(traj_folder, traj_file)
            trajectory_df = pd.read_csv(traj_file_path, header=None, 
                                        names=['Latitude', 'Longitude', 'unused1', 'Altitude', 'unused2', 'Date', 'Time'],
                                        parse_dates=[['Date', 'Time']],
                                        skiprows = 6
                                    )
            # Filter by study bbox
            trajectory_df = trajectory_df[
                (trajectory_df['Latitude'] >= study_min_lat) &
                (trajectory_df['Latitude'] <= study_max_lat) &
                (trajectory_df['Longitude'] >= study_min_lon) &
                (trajectory_df['Longitude'] <= study_max_lon)]
            
            # Filter the points within the specified study time range
            trajectory_df = trajectory_df[
                (trajectory_df['Date_Time'] >= study_start_time) & (trajectory_df['Date_Time'] <= study_end_time)
            ]

            if len(trajectory_df) == 0:
                continue
            trajectory_df['epoch_ts'] = (trajectory_df['Date_Time'].astype(int) / 10**9)
            
            

            # Find overlapping labels for the current trajectory
            for _, label_row in labels_df.iterrows():
                label_start_time = label_row['Start Time']
                label_end_time = label_row['End Time']
                label_mode = label_row['Transportation Mode']
                
                # Check if the current trajectory overlaps with the label
                if (
                    ((trajectory_df['Date_Time'].min() >= label_start_time) and (trajectory_df['Date_Time'].min() <= label_end_time)) or
                    ((trajectory_df['Date_Time'].max() >= label_start_time) and (trajectory_df['Date_Time'].max() <= label_end_time)) or
                    ((trajectory_df['Date_Time'].min() <= label_start_time) and (trajectory_df['Date_Time'].max() >= label_end_time))
                ):
                    #overlapping_trajectories.append((traj_file, label_mode, label_start_time, label_end_time, user_id))
                    #print(overlapping_trajectories)
                    #trajectory_df['epoch_ts'] = (trajectory_df['Date_Time'].astype(int) / 10**9)
                    
                    # filter the part that matches this label's range
                    trajectory_df = trajectory_df[
                        (trajectory_df['Date_Time'] >= label_start_time) & (trajectory_df['Date_Time'] <= label_end_time)
                    ]
                    trajectory_df['uid'] = user_id
                    points_list = trajectory_df[['uid', 'Longitude', 'Latitude', 'epoch_ts']].values.tolist()
                    all_points.extend(points_list)
    print(len(all_points))
    latitudes = [lat for _, lat, _, _ in all_points]
    longitudes = [lon for _, _, lon, _ in all_points]
    uids = [idd for idd, _, _, _ in all_points]

    print(len(set(uids)))

    # Calculate bbox using min and max functions
    bbox = (min(latitudes), min(longitudes), max(latitudes), max(longitudes))

    print(bbox)

    columns = ['id', 'longitude', 'latitude', 'timestamp']
    all_points_df = pd.DataFrame(all_points, columns=columns)
    all_points_df['timestamp'] = all_points_df['timestamp'].astype(int)
    all_points_df = all_points_df.sort_values(by='timestamp')

    all_points_df.to_csv('data/filtered_points_w_header.csv', header=True, index=False)
    all_points_df.to_csv('data/filtered_points_wo_header.csv', header=False, index=False)
                    #break
        #break
            
        #     traj_filenames.append(traj_file)
        #     traj_start_times.append(trajectory_df['Date_Time'].min())
        #     traj_end_times.append(trajectory_df['Date_Time'].max())
        
        # trajs_df = pd.DataFrame({
        #     'filename': traj_filenames,
        #     'traj_start_time': traj_start_times,
        #     'traj_end_time': traj_end_times
        # })
        

        # # Iterate through each label in labels_df
        # for _, label_row in labels_df.iterrows():
        #     label_start_time = label_row['Start Time']
        #     label_end_time = label_row['End Time']
        #     label_mode = label_row['Transportation Mode']

        #     # Filter trajs_df to find overlapping trajectories with the current label
        #     overlapping_traj_mask = (
        #         ((trajs_df['traj_start_time'] >= label_start_time) & (trajs_df['traj_start_time'] <= label_end_time)) |
        #         ((trajs_df['traj_end_time'] >= label_start_time) & (trajs_df['traj_end_time'] <= label_end_time)) |
        #         ((trajs_df['traj_start_time'] <= label_start_time) & (trajs_df['traj_end_time'] >= label_end_time))
        #     )

        #     overlapping_filenames = trajs_df.loc[overlapping_traj_mask, 'filename'].tolist()
        #     if len(overlapping_filenames) > 1:
        #         print(overlapping_filenames)
        #         break
        #     overlapping_trajectories.extend([
        #         (filename, label_mode, label_start_time, label_end_time, user_id) 
        #             for filename in overlapping_filenames
        #     ])



if __name__ == '__main__':
    main()
