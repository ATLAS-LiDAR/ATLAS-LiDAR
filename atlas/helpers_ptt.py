import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import torch
import copy


def print_dict(dict_):
    for key in dict_.keys():
        item =  dict_[key]
        if type(item) == int or type(item) == float or type(item) == str or type(item) == bool or type(item) == dict:
            shape = 1
        else:
            shape = item.shape
        print(key, "| shape: ",shape , "| type: ", type(item))
        
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def get_stats_for_range(r, stats):
    for (rmin, rmax), values in stats.items():
        if rmin <= r < rmax:
            return values
    return None
def plot_4frame_sequence(points):
    """
    points: Nx6 array (x,y,z,*,*,lag)
    """

    pio.renderers.default = "iframe_connected"
    # pio.renderers.default = "browser"


    frame_points = []
    mins = []
    maxs = []

    # Collect points for each lag
    for lag in reversed(range(4)):        
        pts = isolate_frame_points(points, lag)[:, :3]
        print(f"points shape inside plot function: {pts.shape[0]}")
        frame_points.append(pts)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))

    global_min = np.min(np.vstack(mins), axis=0)
    global_max = np.max(np.vstack(maxs), axis=0)

    def make_scatter(pts):
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=pts[:, 2],
                colorscale='Viridis',
                cmin=global_min[2],
                cmax=global_max[2],
                opacity=0.7
            )
        )

    frames = [
        go.Frame(
            data=[make_scatter(frame_points[i])],
            name=str(i)
        )
        for i in range(4)
    ]

    fig = go.Figure(
        data=[make_scatter(frame_points[0])],
        frames=frames
    )

    fig.update_layout(
        title="4-Frame CenterPoint Multiframe Sequence",
        scene=dict(
            xaxis=dict(range=[global_min[0], global_max[0]]),
            yaxis=dict(range=[global_min[1], global_max[1]]),
            zaxis=dict(range=[global_min[2], global_max[2]]),
            aspectmode='data'
        ),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 600, "redraw": True}}]
                }
            ]
        }]
    )

    fig.show()

def avg_spoof_range(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    return(np.mean(r))

def cart_2_spherical(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    theta = theta % (2 * np.pi)
    arg = z/r
    phi = np.arcsin(arg)
    return r, theta, phi

# def plot_trace(trace):
#     x = trace[:, 0]
#     y = trace[:, 1]
#     z = trace[:, 2]
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     # Plot the data points
#     ax.scatter(x, y, z) 

#     # Label the axes
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     ax.set_title('Sparse object example')
#     # --- key part ---
#     dx = x.max() - x.min()
#     dy = y.max() - y.min()
#     dz = z.max() - z.min()

#     ax.set_box_aspect((dx, dy, dz))  # rectangular prism
#     # -----------------

#     plt.show()


def plot_trace(trace):
    pio.renderers.default = "iframe_connected"
    x = trace[:, 0]
    y = trace[:, 1]
    z = trace[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, 
        y=y, 
        z=z,
        mode='markers',
        marker=dict(
            size=3,              # Adjust size for visibility
            color=z,             # Optional: Color by height (Z) for depth perception
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Update layout to match your matplotlib logic
    fig.update_layout(
        title='Sparse object example',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            # --- Key Part ---
            # aspectmode='data' automatically handles the dx, dy, dz calculation 
            # you were doing manually. It ensures 1 unit in X looks the same 
            # physical length as 1 unit in Y or Z.
            aspectmode='data' 
            # ----------------
        ),
        margin=dict(r=0, l=0, b=0, t=40) # Tight layout
    )

    fig.show()
    
    
def isolate_frame_points(pts, lag):

    # 2. Force a copy to ensure memory is contiguous
    pts_clean = np.array(pts, copy=True)
    # 3. Create a strict mask on the last column (index 5)
    # Using a gap-based threshold (0.05) since lags are 0, 0.1, 0.2, 0.3
    lags = np.round(pts_clean[:, -1], decimals=1)
    target_lag = 0.1*lag
    # mask = np.abs(pts_clean[:, 5]) < 0.05
    mask = (lags == target_lag)
    # 4. Apply
    current_frame = pts_clean[mask]
    # print(f"Verified Max Lag: {current_frame[:, 5].max()}")
    return current_frame
    
def plot_waymo_frame(points):
    """
    points: Nx6 array (x, y, z, *, *, lag)
    Only the first 3 columns are used for plotting.
    """

    # import plotly.graph_objects as go
    # import plotly.io as pio
    # import numpy as np

    pio.renderers.default = "iframe_connected"
    # pio.renderers.default = "browser"

    pts = points[:, :3]
    # print(f"points in frame: {pts.shape[0]}")

    # Compute bounds for consistent axis scaling
    global_min = pts.min(axis=0)
    global_max = pts.max(axis=0)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=pts[:, 2],
                    colorscale="Viridis",
                    cmin=global_min[2],
                    cmax=global_max[2],
                    opacity=0.7,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Waymo LiDAR Frame",
        scene=dict(
            xaxis=dict(range=[global_min[0], global_max[0]]),
            yaxis=dict(range=[global_min[1], global_max[1]]),
            zaxis=dict(range=[global_min[2], global_max[2]]),
            aspectmode="data",
        ),
    )

    fig.show()
    

    
    
    
# def plot_frame_cp_mf(base_pts, max_points, base_frame, highlight_counts):
    
    
#      """
#         plot 4 frames from waymo multiframe
#          :param highlight_counts: List of 4 integers [n0, n1, n2, n3]
#      """
    
#     pio.renderers.default = "iframe_connected"
#     rng = np.random.default_rng(0)
#     frame_data_list = [] # Will store (normal_pts, special_pts) tuples
#     mins, maxs = [], []
        
#     for i in range(0, 4):
#         current_frame = isolate_frame_points(base_pts, (3 - i))
#         pts = current_frame[:, :3]
#         n_highlight = highlight_counts[i]
        
#         # Explicitly separate the two groups
#         normal_pts = pts[:-n_highlight] if n_highlight > 0 else pts
#         special_pts = pts[-n_highlight:] if n_highlight > 0 else np.array([]).reshape(0, 3)

#         # Downsample only the background scene if needed
#         if normal_pts.shape[0] > (max_points - n_highlight):
#             sel = rng.choice(normal_pts.shape[0], size=max_points - n_highlight, replace=False)
#             normal_pts = normal_pts[sel]

#         frame_data_list.append((normal_pts, special_pts))
        
#         # Update bounds for the background scene
#         if normal_pts.size > 0:
#             mins.append(normal_pts.min(axis=0))
#             maxs.append(normal_pts.max(axis=0))

#     global_min = np.min(np.vstack(mins), axis=0)
#     global_max = np.max(np.vstack(maxs), axis=0)

#     # Helper to create the two traces per frame
#     def get_traces(normal, special):
#     # Background Lidar Trace (Always present)
#         bg_trace = go.Scatter3d(
#             x=normal[:, 0], y=normal[:, 1], z=normal[:, 2],
#             mode='markers',
#             name='Lidar Scene',
#             marker=dict(
#                 size=1,
#                 color=normal[:, 2],
#                 colorscale='Viridis',
#                 cmin=global_min[2],
#                 cmax=global_max[2],
#                 opacity=0.6
#             )
#         )

#         # Spoof Points Trace
#         # If special is empty (n=0), this trace exists but is invisible/empty
#         spoof_trace = go.Scatter3d(
#             x=special[:, 0] if special.size > 0 else [None],
#             y=special[:, 1] if special.size > 0 else [None],
#             z=special[:, 2] if special.size > 0 else [None],
#             mode='markers',
#             name='Spoof Points',
#             marker=dict(
#                 size=4, 
#                 color='red',
#                 opacity=1.0 if special.size > 0 else 0.0 # Fully transparent if n=0
#             )
#         )
#         return [bg_trace, spoof_trace]

#     # Initialize with the first frame
#     initial_traces = get_traces(*frame_data_list[0])
    
#     # Create animation frames (each frame now has 2 data objects)
#     frames = [
#         go.Frame(data=get_traces(n, s), name=str(i))
#         for i, (n, s) in enumerate(frame_data_list)
#     ]

#     fig = go.Figure(data=initial_traces, frames=frames)

#     fig.update_layout(
#         title=f'Waymo sequence frames {base_frame-3}-{base_frame}',
#         scene=dict(
#             xaxis=dict(range=[global_min[0], global_max[0]], title='x'),
#             yaxis=dict(range=[global_min[1], global_max[1]], title='y'),
#             zaxis=dict(range=[global_min[2], global_max[2]], title='z'),
#             aspectmode='data',
#         ),
#         showlegend=True, # Added legend so you can toggle spoof points on/off
#         margin=dict(l=0, r=0, t=30, b=0),
#         updatemenus=[{
#             'type': 'buttons',
#             'buttons': [
#                 {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}}]},
#                 {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}}]}
#             ]
#         }],
#         sliders=[{
#             'steps': [{'method': 'animate', 'label': str(i), 'args': [[str(i)], {'frame': {'duration': 5, 'redraw': True}}]} for i in range(4)]
#         }]
#     )

#     fig.show()    
    
    
    
    
    
    
    

    
    
    
    
    
# def plot_frame_cp_mf(base_pts, max_points, base_frame, highlight_counts):
    
#     """
#         plot 4 frames from waymo multiframe
#         :param highlight_counts: List of 4 integers [n0, n1, n2, n3]
#     """
    
#     pio.renderers.default = "iframe_connected"
#     rng = np.random.default_rng(0)

#     frame_points = []
#     mins = []
#     maxs = []
#     # base_pts = dataset[base_frame]['points']
#     # print(base_pts.shape)
#     for i in range(0, 4):
  
#         current_frame = isolate_frame_points(base_pts, (3-i))
#         pts = current_frame[:, :3]
 
# #         if pts.shape[0] > max_points:
# #             # sel = rng.choice(pts.shape[0], size=max_points, replace=False)
# #             # pts = pts[sel]
            
# #             # Note: We take the first (N - highlight) points and sample from them, 
# #             # then append the last 'highlight' points to ensure they remain visible.
# #             h_count = highlight_counts[i]
# #             main_body = pts[:-h_count]
# #             highlight_part = pts[-h_count:]
            
# #             if main_body.shape[0] > (max_points - h_count):
# #                 sel = rng.choice(main_body.shape[0], size=max_points - h_count, replace=False)
# #                 main_body = main_body[sel]
            
# #             pts = np.vstack([main_body, highlight_part])
            
#         # print(pts.shape)
#         frame_points.append(pts)
#         mins.append(pts.min(axis=0))
#         maxs.append(pts.max(axis=0))

#     global_min = np.min(np.vstack(mins), axis=0)
#     global_max = np.max(np.vstack(maxs), axis=0)

#     def make_scatter(pts, n_highlight):
#         # return go.Scatter3d(
#         #     x=pts[:, 0],
#         #     y=pts[:, 1],
#         #     z=pts[:, 2],
#         #     mode='markers',
#         #     marker=dict(size=1, color=pts[:, 2], colorscale='Viridis', cmin=global_min[2],
#         #         cmax=global_max[2], opacity=0.6),
#         # )
# # 1. Create the color array based on Z-height
#         color_vals = np.copy(pts[:, 2])
        
#         # 2. Assign a 'special' value to the last n points. 
#         # We use a value slightly above the max Z to trigger the end of our custom scale.
#         if n_highlight > 0:
#             color_vals[-n_highlight:] = global_max[2] + 2.0

#         return go.Scatter3d(
#             x=pts[:, 0],
#             y=pts[:, 1],
#             z=pts[:, 2],
#             mode='markers',
#             marker=dict(
#                 size=1.2, # Slightly larger for visibility
#                 color=color_vals,
#                 # 3. Custom Colorscale: Map 0-95% to Viridis, and the top 5% to Red
#                 colorscale=[
#                     [0.0, '#440154'],   # Viridis Purple
#                     [0.45, '#21918c'],  # Viridis Teal
#                     [0.90, '#fde725'],  # Viridis Yellow
#                     [0.91, 'red'],      # Start of Highlight Red
#                     [1.0, 'red']        # End of Highlight Red
#                 ],
#                 # 4. Set the range so the 'normal' points stay within the Viridis part
#                 cmin=global_min[2],
#                 cmax=global_max[2] + 2.0, 
#                 opacity=0.7
#             ),
#         )

#     frames = [
#         go.Frame(data=[make_scatter(pts, highlight_counts[i])], name=str(i))
#         for i, pts in zip(range(0, 4), frame_points)
#     ]


#     fig = go.Figure(
#         data=[make_scatter(frame_points[0], highlight_counts[i])],
#         frames=frames,
#     )

#     fig.update_layout(
#         title=f'Waymo sequence frames {base_frame-3}-{base_frame}',
#         scene=dict(
#             xaxis=dict(range=[global_min[0], global_max[0]], title='x'),
#             yaxis=dict(range=[global_min[1], global_max[1]], title='y'),
#             zaxis=dict(range=[global_min[2], global_max[2]], title='z'),
#             aspectmode='data',
#         ),
#         margin=dict(l=0, r=0, t=30, b=0),
#         updatemenus=[{
#             'type': 'buttons',
#             'showactive': True,
#             'x': 0.1,
#             'y': 0,
#             'pad': {'r': 10, 't': 70},
#             'buttons': [
#                 {
#                     'label': 'Play',
#                     'method': 'animate',
#                     'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
#                 },
#                 {
#                     'label': 'Pause',
#                     'method': 'animate',
#                     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
#                 },
#             ],
#         }],
#         sliders=[{
#             'x': 0.1,
#             'y': 0,
#             'len': 0.9,
#             'steps': [
#                 {
#                     'method': 'animate',
#                     'args': [[str(base_frame + (i-3))], {'frame': {'duration': 5, 'redraw': True}, 'mode': 'immediate'}],
#                     'label': str(base_frame + (i-3)),
#                 }
#                 for i in range(0, 4)
#             ],
#         }],
#     )

#     fig.show()
#     fig.write_html("sequence_animation.html")

    
    
    
def plot_frame(data_dict, dataset, spoof_frame, start_idx, end_idx, max_points):
    pio.renderers.default = "iframe_connected"

    # Animate frames 0-197 from the dataset (same sequence assumed)
    # start_idx = 0
    # end_idx = 1
    # max_points = 150000  # downsample per frame for speed; raise/lower as needed
    rng = np.random.default_rng(0)

    frame_points = []
    mins = []
    maxs = []

    for i in range(start_idx, end_idx + 1):
        # info = dataset.infos[i]
        # seq = info['point_cloud']['lidar_sequence']
        # sample_idx = info['point_cloud']['sample_idx']
        if(i == spoof_frame):
            pts = data_dict['points'][:, :3]
        else:
            pts = dataset[i]['points'][:, :3]
            
        # print(i, pts.shape)
        if pts.shape[0] > max_points:
            sel = rng.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[sel]

        frame_points.append(pts)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))

    global_min = np.min(np.vstack(mins), axis=0)
    global_max = np.max(np.vstack(maxs), axis=0)

    def make_scatter(pts):
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=1, color=pts[:, 2], colorscale='Viridis', cmin=global_min[2],
                cmax=global_max[2], opacity=0.6),
        )

    frames = [
        go.Frame(data=[make_scatter(pts)], name=str(i))
        for i, pts in zip(range(start_idx, end_idx + 1), frame_points)
    ]


    fig = go.Figure(
        data=[make_scatter(frame_points[0])],
        frames=frames,
    )

    fig.update_layout(
        title=f'Waymo sequence frames {start_idx}-{end_idx}',
        scene=dict(
            xaxis=dict(range=[global_min[0], global_max[0]], title='x'),
            yaxis=dict(range=[global_min[1], global_max[1]], title='y'),
            zaxis=dict(range=[global_min[2], global_max[2]], title='z'),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'x': 0.1,
            'y': 0,
            'pad': {'r': 10, 't': 70},
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                },
            ],
        }],
        sliders=[{
            'x': 0.1,
            'y': 0,
            'len': 0.9,
            'steps': [
                {
                    'method': 'animate',
                    'args': [[str(i)], {'frame': {'duration': 5, 'redraw': True}, 'mode': 'immediate'}],
                    'label': str(i),
                }
                for i in range(start_idx, end_idx + 1)
            ],
        }],
    )

    fig.show()
    



def plot_single_frame(pts, max_points=150000, title="Point Cloud Visualization"):
    """
    Plots a single 3D point cloud frame.
    :param pts: Nx3 or Nx6 numpy array. We only use the first 3 columns.
    :param max_points: Downsampling threshold for performance.
    :param title: Plot title.
    """
    # Ensure we only use X, Y, Z
    pio.renderers.default = "iframe_connected"

    pts_to_plot = pts[:, :3]
    
    # 1. Downsampling Logic
    if pts_to_plot.shape[0] > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(pts_to_plot.shape[0], size=max_points, replace=False)
        pts_to_plot = pts_to_plot[sel]

    # 2. Axis/Bounding Box Logic
    # We calculate mins and maxs to ensure the camera doesn't squash the scene
    mins = pts_to_plot.min(axis=0)
    maxs = pts_to_plot.max(axis=0)

    # 3. Create the Scatter object
    scatter = go.Scatter3d(
        x=pts_to_plot[:, 0],
        y=pts_to_plot[:, 1],
        z=pts_to_plot[:, 2],
        mode='markers',
        marker=dict(
            size=1, 
            color=pts_to_plot[:, 2], # Color by height (Z-axis)
            colorscale='Viridis', 
            opacity=0.6
        ),
    )

    fig = go.Figure(data=[scatter])

    # 4. Layout & Aspect Ratio
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[mins[0], maxs[0]], title='X (m)'),
            yaxis=dict(range=[mins[1], maxs[1]], title='Y (m)'),
            zaxis=dict(range=[mins[2], maxs[2]], title='Z (m)'),
            # 'data' aspectmode is crucial for point clouds to look "real"
            aspectmode='data', 
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.show()
    
    
#     azimuth_elevation_indices_trace = []
# dtheta = np.deg2rad(0.1358)
# grid = np.zeros((len(azimuths), len(elevations)))
# for i in range (0, len(trace_spherical)):
#     # Your specific value
#     query_value = trace_spherical[i][2] 

#     # Calculate absolute difference between the query and ALL elevation values
#     differences = np.abs(elevations - query_value)

#     # Find the index of the smallest difference
#     closest_index = np.argmin(differences)

#     # print(f"Closest Index: {closest_index}")
#     # print(f"Value at Index: {elevations[closest_index]}")
#     # print(np.floor((trace_spherical[i][1] % (2*np.pi)) / dtheta), closest_index)
#     azimuth_elevation_indices_trace.append((np.floor((trace_spherical[i][1] % (2*np.pi)) / dtheta), closest_index))

# 1. Define mapping helper
# Waymo labels: 1=Vehicle, 2=Pedestrian, 3=Cyclist
# dataset.class_names is typically ['Vehicle', 'Pedestrian', 'Cyclist']
def inject_gt_names(data_dict, class_names):
    if 'gt_boxes' in data_dict and 'gt_names' not in data_dict:
        # Extract the last column (label index)
        # gt_boxes shape is (N, 8), index 7 is the label
        labels = data_dict['gt_boxes'][:, -1].astype(int)
        
        # Map index to name (Label 1 -> Index 0)
        # We use l-1 because Waymo labels are 1-based
        names = np.array([class_names[l - 1] for l in labels])
        data_dict['gt_names'] = names
    return data_dict

def convert_to_batch(dict_):
    data_dict = copy.deepcopy(dict_)
    device = dict_['points'].device
    data_dict['sample_idx'] = torch.tensor([dict_['sample_idx']], device = device)
    data_dict['poses'] = dict_['poses'].unsqueeze(0)
    data_dict['roi_boxes'] = dict_['roi_boxes'].unsqueeze(0)
    data_dict['roi_scores'] = dict_['roi_scores'].unsqueeze(0)
    data_dict['roi_labels'] = dict_['roi_labels'].unsqueeze(0)
    data_dict['frame_id'] = [dict_['frame_id']]
    data_dict['gt_boxes'] = dict_['gt_boxes'].unsqueeze(0)
    data_dict['use_lead_xyz'] = torch.tensor([1], device = device) if dict_['use_lead_xyz'] else torch.tensor([0], device = device)
    data_dict['metadata'] = [dict_['metadata']]
    
    
    return data_dict

def convert_to_batch_cp_sf(dict_):
    data_dict = copy.deepcopy(dict_)
    device = dict_['points'].device
    data_dict['sample_idx'] = torch.tensor([dict_['sample_idx']], device = device)
    rows_points = dict_['points'].shape[0]
    zeros_points =torch.zeros((rows_points, 1), device = device)
    data_dict['points'] = torch.cat((zeros_points, dict_['points']), dim = 1)
    data_dict['frame_id'] = [dict_['frame_id']]
    data_dict['gt_boxes'] = dict_['gt_boxes'].unsqueeze(0)
    # data_dict['lidar_aug_matrix'] = dict_['lidar_aug_matrix'].unsqueeze(0)
    data_dict['use_lead_xyz'] = torch.tensor([1], device = device) if dict_['use_lead_xyz'] else torch.tensor([0], device = device)
    rows_vc = dict_['voxel_coords'].shape[0]
    zeros_vc = torch.zeros((rows_vc, 1), device = device)
    data_dict['voxel_coords'] = torch.cat((zeros_vc, dict_['voxel_coords']), dim = 1)
    data_dict['metadata'] = [dict_['metadata']]
    
    return data_dict

def convert_to_batch_cp_mf(dict_):
    data_dict = copy.deepcopy(dict_)
    device = dict_['points'].device
    data_dict['sample_idx'] = torch.tensor([dict_['sample_idx']], device = device)
    
    data_dict['frame_id'] = [dict_['frame_id']]
    data_dict['gt_boxes'] = dict_['gt_boxes'].unsqueeze(0)
    # data_dict['lidar_aug_matrix'] = dict_['lidar_aug_matrix'].unsqueeze(0)
    data_dict['use_lead_xyz'] = torch.tensor([1], device = device) if dict_['use_lead_xyz'] else torch.tensor([0], device = device)
    rows_vc = dict_['voxel_coords'].shape[0]
    zeros_vc = torch.zeros((rows_vc, 1), device = device)
    data_dict['voxel_coords'] = torch.cat((zeros_vc, dict_['voxel_coords']), dim = 1)
    data_dict['metadata'] = [dict_['metadata']]
    
    data_dict['poses'] = dict_['poses'].unsqueeze(0)
    
    return data_dict

def cosine_similarity(a, b, eps=1e-8):
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + eps)


# from pcdet.config import cfg, cfg_from_yaml_file
# cfg.clear()

# CFG_FILE_SF = '../OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml' 
# CKPT_SF = '../OpenPCDet/output/cfgs/custom_models/centerpoint_singleframe_waymo/default/ckpt/checkpoint_epoch_30.pth'  # <- put your ckpt here

# logger = common_utils.create_logger()
# logger.info(f'Loaded cfg from {CFG_FILE_SF}')

# cfg_from_yaml_file(
#     CFG_FILE_SF,
#     cfg
# )

# cfg.TAG = 'centerpoint_singleframe'
# cfg.EXP_GROUP_PATH = 'waymo_singleframe'

# dataset_sf, test_loader_sf, _ = build_dataloader(
#     dataset_cfg=cfg.DATA_CONFIG,
#     class_names=cfg.CLASS_NAMES,
#     batch_size=1,
#     dist=False,
#     workers=4,
#     logger=logger,
#     training=False
# )

# len_test = len(dataset)
# logger.info(f'Test set length: {len_test}')
# cp_sf_model = build_network(
#     model_cfg=cfg.MODEL,
#     num_class=len(cfg.CLASS_NAMES),
#     dataset=dataset_sf
# )

# logger.info(f'Loading checkpoint from: {CKPT_SF}')
# cp_sf_model.load_params_from_file(filename=CKPT_SF, logger=logger, to_cpu=False)
# cp_sf_model.cuda()
# cp_sf_model.eval()

# data_iter_sf = iter(test_loader_sf)
# batch_dict_sf = next(data_iter_sf)

# load_data_to_gpu(batch_dict_sf)





#code to figure out approximate distance travelled from frame - 31--> frame

# frame = 32
# infos = dataset_mf.infos
# for frame in range(31, 197):
#     first = None
#     last = None
#     for i in range(frame-31, frame+1):
#         if i==frame-31: first=infos[i]['pose'][:,-1][:3]
#         if i==frame: last=infos[i]['pose'][:,-1][:3]
#         # print(infos[i]['pose'][:,-1][:3])
#     print(np.linalg.norm(first-last))


# lag_0 = isolate_frame_points(dataset_mf[frame]['points'], 0)
# lag_1 = isolate_frame_points(dataset_mf[frame]['points'], 1)
# lag_2 = isolate_frame_points(dataset_mf[frame]['points'], 2)
# lag_3 = isolate_frame_points(dataset_mf[frame]['points'], 3)
# np.save('lag_seperated_data', lag_0)
# np.save('lag_seperated_data', lag_1)
# np.save('lag_seperated_data', lag_2)
# np.save('lag_seperated_data', lag_3)



#!trace collection code


# traces_smaller = []

# flag = 0
# for i in range(0, 100):
#     boxes = dataset[i]['gt_boxes']
#     points = dataset[i]['points']
#     mask = boxes[:, 7]==1
#     boxes_car = boxes[mask]

#     extracted_traces = []
#     for box in boxes_car:
#         cx, cy, cz = box[0], box[1], box[2]
#         dx, dy, dz = box[3], box[4], box[5]
#         heading = box[6]

#         #put box center at origin
#         shifted_points = points[:, :3] - np.array([cx, cy, cz])

#         #rotate points to align with the box's local axes
#         #rotate by -heading to undo boxes rotation
#         cos_a = np.cos(-heading)
#         sin_a = np.sin(-heading)
#         R = np.array([
#             [cos_a, -sin_a, 0],
#             [sin_a,  cos_a, 0],
#             [0,      0,     1]])
#         local_points = shifted_points@R.T
#         inside_mask = (
#             (np.abs(local_points[:, 0]) <= dx / 2) &
#             (np.abs(local_points[:, 1]) <= dy / 2) &
#             (np.abs(local_points[:, 2]) <= dz / 2)
#         )
#         points_inside = points[inside_mask]
#         if len(points_inside) > 0:
#             extracted_traces.append({
#                 'frame': i,
#                 'box': box,
#                 'points': points_inside,
#                 'num_points': len(points_inside)
#             })
#     # print(f"=============frame {i}============")
#     for trace in extracted_traces:
#     # print(trace['points'])
#         # print(trace['num_points'], np.linalg.norm(trace['box'][:3]))
#         if(trace['num_points'] <= 100 and trace['num_points'] >=30): traces_smaller.append(trace)
#         if(len(traces_smaller) == 100):
#             flag = 1
#             break
#     if(flag): break

#     # if(len(traces) == 100): break
    
# traces_bigger = []
# flag = 0
# for i in range(0, 100):
#     boxes = dataset[i]['gt_boxes']
#     points = dataset[i]['points']
#     mask = boxes[:, 7]==1
#     boxes_car = boxes[mask]

#     extracted_traces = []
#     for box in boxes_car:
#         cx, cy, cz = box[0], box[1], box[2]
#         dx, dy, dz = box[3], box[4], box[5]
#         heading = box[6]

#         #put box center at origin
#         shifted_points = points[:, :3] - np.array([cx, cy, cz])

#         #rotate points to align with the box's local axes
#         #rotate by -heading to undo boxes rotation
#         cos_a = np.cos(-heading)
#         sin_a = np.sin(-heading)
#         R = np.array([
#             [cos_a, -sin_a, 0],
#             [sin_a,  cos_a, 0],
#             [0,      0,     1]])
#         local_points = shifted_points@R.T
#         inside_mask = (
#             (np.abs(local_points[:, 0]) <= dx / 2) &
#             (np.abs(local_points[:, 1]) <= dy / 2) &
#             (np.abs(local_points[:, 2]) <= dz / 2)
#         )
#         points_inside = points[inside_mask]
#         if len(points_inside) > 0:
#             extracted_traces.append({
#                 'frame': i,
#                 'box': box,
#                 'points': points_inside,
#                 'num_points': len(points_inside)
#             })
#     # print(f"=============frame {i}============")
#     for trace in extracted_traces:
#     # print(trace['points'])
#         # print(trace['num_points'], np.linalg.norm(trace['box'][:3]))
#         if(trace['num_points'] <= 200 and trace['num_points'] >=100): traces_bigger.append(trace)
#         if(len(traces_bigger) == 100):
#             flag = 1
#             break
#     if(flag): break

# # for i, trace in enumerate(traces):
# #     print(i, trace['frame'], trace['num_points'])

# traces = traces_smaller + traces_bigger
# sorted_traces = sorted(traces, key=lambda item: item['num_points'])

# # 3. Open the file in binary write mode and dump the data
# file_path = 'traces.pkl'
# try:
#     with open(file_path, 'wb') as file:
#         pkl.dump(sorted_traces, file)
#     print(f"Successfully wrote data to '{file_path}'")
# except IOError as e:
#     print(f"Error writing file: {e}")