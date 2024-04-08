import warnings
import math
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from geopy import distance
import folium
import osmnx as ox
import networkx as nx
from shapely.geometry import Polygon
from shapely.geometry import LineString, Point
import random
from geopy.distance import geodesic
import streamlit as st
from streamlit_folium import folium_static
import base64
from io import BytesIO

def process_uploaded_files(uploaded_files):
    dataframes = {}
    data = None

    for idx, file in enumerate(uploaded_files):
        df = pd.read_excel(file)

        # Get the filename without extension
        filename_without_extension = file.name.split('.')[0]

        # Assign dataframe to dictionary using filename as key
        dataframes[filename_without_extension] = df

        # Assign specific dataframes
        if idx == 0:
            data = df.copy()

    return dataframes, data

def create_square_map(group_1_df, min_lat, max_lat, min_lon, max_lon):
    # Tính toán tọa độ của tâm hình vuông
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Tạo bản đồ mới với tọa độ tâm làm trung tâm và phóng to sao cho hình vuông nằm hoàn toàn trong bản đồ
    baby_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Thêm marker cho mỗi điểm trong group_1_df
    for index, row in group_1_df.iterrows():
        popup_content = f"Order: {index+1}<br>OutletID: {row['OutletID']}<br>OutletName: {row['OutletName']}<br>Latitude: {row['Latitude']}<br>Longitude: {row['Longitude']}"
        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=folium.Popup(popup_content, max_width=300)).add_to(baby_map)

    # Chọn màu ngẫu nhiên từ danh sách các màu
    colors = ['black', 'beige', 'lightblue', 'gray', 'blue', 'darkred', 'lightgreen', 'purple', 'red', 'green', 'lightred', 'darkblue', 'darkpurple', 'cadetblue', 'orange', 'pink', 'lightgray', 'darkgreen', 'pink', 'yellow', 'purple']

    random_color = random.choice(colors)

    # Tạo hình vuông bao quanh các điểm với màu được chọn ngẫu nhiên
    folium.Rectangle(bounds=[(min_lat, min_lon), (max_lat, max_lon)], color=random_color, fill=True, fill_opacity=0.2).add_to(baby_map)

    return baby_map

def remove_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def find_nearest_corner_point(min_lat, min_lon, max_lat, max_lon, group_1_df):
    corners = [(min_lat, min_lon), (min_lat, max_lon), (max_lat, min_lon), (max_lat, max_lon)]

    nearest_points = {}

    for corner in corners:
        min_distance = np.inf
        nearest_point = None

        for index, row in group_1_df.iterrows():
            distance = np.sqrt((corner[0] - row['Latitude'])**2 + (corner[1] - row['Longitude'])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = row

        nearest_points[corner] = {'point': nearest_point, 'distance': min_distance}

    min_distance_data = min(nearest_points.items(), key=lambda x: x[1]['distance'])

    return min_distance_data

def create_final_df(filtered_df, min_distance_data, no_outlet):
    # Khởi tạo final_df1 với cột dữ liệu tương tự như filtered_df
    final_df1 = pd.DataFrame(columns=filtered_df.columns)

    # Thêm điểm gần nhất vào final_df1
    final_df1 = pd.concat([final_df1, min_distance_data[1]['point'].to_frame().T], ignore_index=True)

    # Tính khoảng cách giữa các điểm trong filtered_df và điểm gần nhất
    filtered_df['distance_to_min'] = filtered_df.apply(lambda row: np.linalg.norm(np.array((row['Latitude'], row['Longitude'])) - np.array(min_distance_data[0])), axis=1)

    # Sắp xếp filtered_df theo khoảng cách tới điểm gần nhất
    filtered_df_sorted = filtered_df.sort_values(by='distance_to_min')

    # Tìm điểm kế tiếp ngắn nhất từ min_distance_data[1]['point']
    current_point = min_distance_data[1]['point']

    # Lặp để thêm điểm cho đến khi final_df1 có đủ 30 điểm
    while len(final_df1) < no_outlet:
        # Lấy điểm kế tiếp ngắn nhất
        next_point_index = filtered_df_sorted.index[0]
        next_point = filtered_df_sorted.iloc[0]

        # Loại bỏ điểm đã chọn khỏi filtered_df_sorted
        filtered_df_sorted = filtered_df_sorted.drop(next_point_index)

        # Nếu điểm kế tiếp không trùng với điểm hiện tại, thêm vào final_df1
        if next_point_index != current_point.name:
            final_df1 = pd.concat([final_df1, pd.DataFrame([next_point], columns=final_df1.columns)], ignore_index=True)

        # Cập nhật điểm hiện tại là điểm kế tiếp đã chọn
        current_point = next_point

    return final_df1

def draw_small_square(final_df1, map):
    # Tính toán tọa độ của hình vuông bao quanh các điểm trong final_df
    min_lat_final1 = final_df1['Latitude'].min()
    max_lat_final1 = final_df1['Latitude'].max()
    min_lon_final1 = final_df1['Longitude'].min()
    max_lon_final1 = final_df1['Longitude'].max()

    # Chọn màu ngẫu nhiên từ danh sách các màu
    colors = ['black', 'beige', 'lightblue', 'gray', 'blue', 'darkred', 'lightgreen', 'purple', 'red', 'green', 'lightred', 'white', 'darkblue', 'darkpurple', 'cadetblue', 'orange', 'pink', 'lightgray', 'darkgreen', 'pink', 'yellow', 'purple']
    random_color = random.choice(colors)

    # Tạo hình vuông bao quanh các điểm trong final_df với màu được chọn ngẫu nhiên
    folium.Rectangle(bounds=[(min_lat_final1, min_lon_final1), (max_lat_final1, max_lon_final1)], color=random_color, fill=True, fill_opacity=0.2).add_to(map)

    return map

def euclidean_distance(point1_coords, point2_coords):
    return math.sqrt((point2_coords['Longitude'] - point1_coords['Longitude'])**2 + (point2_coords['Latitude'] - point1_coords['Latitude'])**2)

def calculate_distance_euclidean_two_points(point1_coords, point2_coords, filtered_df):
    distances = {}
    for index, row in filtered_df.iterrows():
        distance = euclidean_distance(point1_coords, {'Longitude': row['Longitude'], 'Latitude': row['Latitude']})
        distance += euclidean_distance(point2_coords, {'Longitude': row['Longitude'], 'Latitude': row['Latitude']})
        distances[index] = distance
    return distances

def create_final_df2(filtered_sorted_df, closest_points, no_oulet):
    final_df2 = pd.DataFrame(columns=filtered_sorted_df.columns)
    final_df2 = pd.concat([final_df2, closest_points.iloc[0].to_frame().T], ignore_index=True)

    # Tính khoảng cách giữa các điểm trong filtered_sorted_df và điểm closest_points[0]
    filtered_sorted_df['distance_to_closest'] = filtered_sorted_df.apply(lambda row: np.linalg.norm(np.array((row['Latitude'], row['Longitude'])) - np.array((closest_points.iloc[0]['Latitude'], closest_points.iloc[0]['Longitude']))), axis=1)

    # Sắp xếp filtered_df theo khoảng cách tới điểm gần nhất
    filtered_distance_sorted = filtered_sorted_df.sort_values(by='distance_to_closest')

    current_point = closest_points.iloc[0]

    while len(final_df2) < no_oulet and current_point is not None:
        # Lấy điểm kế tiếp ngắn nhất
        next_point_index = filtered_distance_sorted.index[0]
        next_point = filtered_distance_sorted.iloc[0]

        # Loại bỏ điểm đã chọn khỏi filtered_distance_sorted
        filtered_distance_sorted = filtered_distance_sorted.drop(next_point_index)

        # Nếu điểm kế tiếp không trùng với điểm hiện tại, thêm vào final_df2
        if next_point_index != current_point.name:
            final_df2 = pd.concat([final_df2, pd.DataFrame([next_point], columns=final_df2.columns)], ignore_index=True)

        # Cập nhật điểm hiện tại là điểm kế tiếp đã chọn
        current_point = next_point

    return final_df2

def Create_square(cleaned_data, no_oulet):
    min_lat = cleaned_data['Latitude'].min()
    max_lat = cleaned_data['Latitude'].max()
    min_lon = cleaned_data['Longitude'].min()
    max_lon = cleaned_data['Longitude'].max()

    new_map = create_square_map(cleaned_data, min_lat, max_lat, min_lon, max_lon)

    min_distance_data = find_nearest_corner_point(min_lat, min_lon, max_lat, max_lon, cleaned_data)

    filtered_df = cleaned_data.drop(min_distance_data[1]['point'].name)
    final_df1 = create_final_df(filtered_df, min_distance_data, no_oulet)

    new_map = draw_small_square(final_df1, new_map)

    i = 1

    final_df1['SRD'] = i

    all_data = pd.DataFrame(columns=final_df1.columns)
    # all_data = all_data.append(final_df1, ignore_index=True)
    all_data = pd.concat([all_data, final_df1], ignore_index=True)

    filtered_sorted_df = filtered_df[~filtered_df['OutletID'].isin(final_df1['OutletID'])]

    i = i + 1

    while not filtered_sorted_df.empty:
        # print(i)
        last_two_rows = final_df1.tail(2)
        # Kiểm tra kích thước của filtered_sorted_df
        if len(filtered_sorted_df) <= 30:
            filtered_sorted_df['SRD'] = i
            # all_data = all_data.append(filtered_sorted_df, ignore_index=True)
            all_data = pd.concat([all_data, filtered_sorted_df], ignore_index=True)
            break

        distances_between_two_points = calculate_distance_euclidean_two_points(last_two_rows.iloc[0], last_two_rows.iloc[1], filtered_sorted_df)
        distances_series = pd.Series(distances_between_two_points)

        closest_points_indices = distances_series.nsmallest(1).index
        closest_points = filtered_sorted_df.loc[closest_points_indices]

        filtered_sorted_df = filtered_sorted_df[~filtered_sorted_df['OutletID'].isin(closest_points['OutletID'])]

        final_df1 = create_final_df2(filtered_sorted_df, closest_points, no_oulet)
        final_df1['SRD'] = i
        # all_data = all_data.append(final_df1, ignore_index=True)
        all_data = pd.concat([all_data, final_df1], ignore_index=True)
        new_map = draw_small_square(final_df1, new_map)

        i = i + 1

        filtered_sorted_df = filtered_sorted_df[~filtered_sorted_df['OutletID'].isin(final_df1['OutletID'])]

    return all_data, new_map

def Create_RD(all_data):
  # Khởi tạo một biến để lưu trữ số của mỗi loại RD
  rd_counts = {}

  # Duyệt qua từng hàng trong DataFrame
  for index, row in all_data.iterrows():
      rd = row['SRD']
      if rd not in rd_counts:
          rd_counts[rd] = 1
      else:
          rd_counts[rd] += 1
      # Tạo giá trị cho cột mới dựa trên RD và số lượng đã đếm
      all_data.at[index, 'RD'] = f"{rd}.{rd_counts[rd]}"

  return all_data

def calculate_distance(point_coords, filtered_df):
    # Convert point_coords to a Point object
    point = Point(point_coords[::-1])  # Reverse the order of coordinates

    # Calculate distance using Shapely and store in a new column
    filtered_df['distance_to_point'] = filtered_df.apply(
        lambda row: point.distance(Point(row['Longitude'], row['Latitude'])),
        axis=1
    )

    # Find the closest point
    closest_point = filtered_df.loc[filtered_df['distance_to_point'].idxmin()]

    return closest_point['distance_to_point'], closest_point

def draw_optimal_path(visited_points, new_map, G, random_color, group_feature):
    # Extract the last two points from visited_points
    visited_points_df = pd.DataFrame(visited_points.tail(2))
    last_point = (visited_points_df.iloc[-2]['Latitude'], visited_points_df.iloc[-2]['Longitude'])
    final_point = (visited_points_df.iloc[-1]['Latitude'], visited_points_df.iloc[-1]['Longitude'])

    # Find the optimal path using OSMnx
    start_node = ox.distance.nearest_nodes(G, last_point[1], last_point[0])
    destination_node = ox.distance.nearest_nodes(G, final_point[1], final_point[0])

    optimal_path_nodes = ox.shortest_path(G, start_node, destination_node, weight='length')

    if optimal_path_nodes is not None:
        optimal_path_coordinates = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in optimal_path_nodes]

        # Vẽ đường dẫn tối ưu giữa hai điểm cuối cùng
        poly_line = folium.PolyLine(optimal_path_coordinates, color=random_color, weight=2.5, opacity=1)
        poly_line.add_to(group_feature)

        # Thêm điểm popup cho điểm thứ hai từ cuối
        last_point_popup = f"Order: {visited_points_df.index[-2] + 1}<br>OutletID: {visited_points_df['OutletID'].iloc[-2]}<br>OutletName: {visited_points_df['OutletName'].iloc[-2]}<br>Latitude: {last_point[0]}<br>Longitude: {last_point[1]}<br>SRD: {visited_points_df['SRD'].iloc[-2]}<br>RD: {visited_points_df['RD'].iloc[-2]}"
        last_marker = folium.Marker(location=[last_point[0], last_point[1]], popup=folium.Popup(last_point_popup, max_width=300))
        last_marker.add_to(group_feature)

        # Thêm điểm popup cho điểm cuối cùng
        final_point_popup = f"Order: {visited_points_df.index[-1] + 1}<br>OutletID: {visited_points_df['OutletID'].iloc[-1]}<br>OutletName: {visited_points_df['OutletName'].iloc[-1]}<br>Latitude: {final_point[0]}<br>Longitude: {final_point[1]}<br>SRD: {visited_points_df['SRD'].iloc[-1]}<br>RD: {visited_points_df['RD'].iloc[-1]}"
        final_marker = folium.Marker(location=[final_point[0], final_point[1]], popup=folium.Popup(final_point_popup, max_width=300))
        final_marker.add_to(group_feature)

    else:
        print("No path found between", start_node, "and", destination_node)

    return new_map

def distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def calculate_distance_between_two_points(point1_coords, point2_coords, graph, filtered_df):
    # Get coordinates of point 1 and point 2
    point1_coords = (point1_coords['Latitude'], point1_coords['Longitude'])
    point2_coords = (point2_coords['Latitude'], point2_coords['Longitude'])

    # Find the closest nodes to point 1 and point 2
    closest_node1 = ox.distance.nearest_nodes(graph, point1_coords[1], point1_coords[0])
    closest_node2 = ox.distance.nearest_nodes(graph, point2_coords[1], point2_coords[0])

    # Find the nearest nodes for all destinations
    destinations_nodes = {}
    for index, row in filtered_df.iterrows():
        dest_node = ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude'])
        destinations_nodes[index] = dest_node

    # Calculate shortest paths for all destinations
    shortest_paths = {}
    for index, dest_node in destinations_nodes.items():
        try:
            distance1 = nx.shortest_path_length(graph, closest_node1, dest_node, weight='length')
            distance2 = nx.shortest_path_length(graph, closest_node2, dest_node, weight='length')
            shortest_paths[index] = distance1 + distance2
        except nx.NetworkXNoPath:
            shortest_paths[index] = float('inf')  # Assign a large value for unreachable nodes

    return shortest_paths

def find_nearest_point(visited_points, closest_points):
    # Lấy ra hai dòng cuối cùng từ visited_points
    last_two_rows = visited_points.tail(2)

    # Tạo tam giác từ hai điểm cuối cùng trong last_two_rows
    points = pd.DataFrame({'Latitude': [last_two_rows.iloc[0]['Latitude'], last_two_rows.iloc[1]['Latitude']],
                           'Longitude': [last_two_rows.iloc[0]['Longitude'], last_two_rows.iloc[1]['Longitude']]})

    # Khởi tạo các biến để lưu thông tin của tam giác nhỏ nhất
    min_perimeter = float('inf')
    min_perimeter_outlet_info = None

    # Duyệt qua mỗi điểm trong closest_points
    for index, row in closest_points.iterrows():
        # Thêm điểm thứ ba vào tam giác
        point_df = pd.DataFrame({'Latitude': [row['Latitude']], 'Longitude': [row['Longitude']]})
        points = pd.concat([points, point_df], ignore_index=True)
        triangle = Polygon(points)

        # Tính chu vi của tam giác
        perimeter = triangle.length

        # So sánh với chu vi nhỏ nhất đã tìm thấy
        if perimeter < min_perimeter:
            min_perimeter = perimeter
            min_perimeter_outlet_info = {'OutletID': row['OutletID'], 'OutletName': row['OutletName'],
                                         'Latitude': row['Latitude'], 'Longitude': row['Longitude'],
                                         'SRD': row['SRD'], 'RD': row['RD']}
        # Xóa điểm thứ ba để chuẩn bị cho lần duyệt tiếp theo
        points = points[:-1]

    # Tạo DataFrame từ min_perimeter_outlet_info
    min_perimeter_outlet_df = pd.concat([pd.DataFrame(min_perimeter_outlet_info, index=[0])])

    return min_perimeter_outlet_df

def create_path_2(group_1_df, G, new_map, random_color, i):
    min_lat = group_1_df['Latitude'].min()
    max_lat = group_1_df['Latitude'].max()
    min_lon = group_1_df['Longitude'].min()
    max_lon = group_1_df['Longitude'].max()

    # Sử dụng hàm để tạo bản đồ
    baby_map = create_square_map(group_1_df, min_lat, max_lat, min_lon, max_lon)
    
    # Sử dụng để tìm điểm gần góc
    min_distance_data = find_nearest_corner_point(min_lat, min_lon, max_lat, max_lon, group_1_df)

    # Lọc data
    filtered_df = group_1_df.drop(min_distance_data[1]['point'].name)
    
    # Assuming min_distance_data is defined somewhere in your code
    nearest_point_coords = min_distance_data[0]

    # Call the function to calculate distances and find the closest point
    closest_distance, closest_point = calculate_distance(nearest_point_coords, filtered_df)

    # print("Closest distance:", closest_distance)
    # print("Closest point:", closest_point)   

    start_point = min_distance_data[1]['point']

    # Tạo DataFrame rỗng để lưu các điểm đã thăm
    visited_points = pd.DataFrame(columns=['OutletID', 'OutletName', 'Latitude', 'Longitude', 'SRD', 'RD'])

    # Thêm start_point vào DataFrame
    visited_points = pd.concat([visited_points, start_point.to_frame().T], ignore_index=True)
    visited_points = pd.concat([visited_points, closest_point.to_frame().T], ignore_index=True)    
    filtered_sorted_df = group_1_df[~group_1_df['OutletID'].isin(visited_points['OutletID'])]

    group_feature = folium.FeatureGroup(name="Group " + str(i)).add_to(new_map)

    new_map = draw_optimal_path(visited_points, new_map, G, random_color, group_feature)
    
    while not filtered_sorted_df.empty:
        last_row = visited_points.tail(1).iloc[0]
        # Tâm của hình tròn
        center_lat, center_lon = last_row['Latitude'], last_row['Longitude']

        # Bán kính ban đầu của hình tròn
        radius = 100

        # Lặp cho đến khi tìm được ít nhất một điểm hoặc không thể tăng bán kính nữa
        while True:
            # Lọc dữ liệu
            filtered_data = []
            for index, row in filtered_sorted_df.iterrows():
                point_lat, point_lon = row['Latitude'], row['Longitude']
                if distance(center_lat, center_lon, point_lat, point_lon) <= radius:
                    filtered_data.append(row)

            # Tạo DataFrame từ dữ liệu lọc
            filtered_df_within_circle = pd.DataFrame(filtered_data)

            # Kiểm tra nếu có ít nhất một điểm trong hình tròn
            if len(filtered_df_within_circle) > 0:
                break
            
            # Nếu không có điểm nào và bán kính đã tăng lên, tăng bán kính thêm 100m và tiếp tục lặp
            radius += 100

        # In ra filtered_df_within_circle
        print(filtered_df_within_circle)

        last_two_rows = visited_points.tail(2)
        distances_between_two_points = calculate_distance_between_two_points(last_two_rows.iloc[0], last_two_rows.iloc[1], G, filtered_df_within_circle)

        closest_points_indices = sorted(distances_between_two_points, key=distances_between_two_points.get)[:2]
        closest_points = filtered_df_within_circle.loc[closest_points_indices]
        min_perimeter_outlet_df = find_nearest_point(visited_points, closest_points)

        visited_points = pd.concat([visited_points, min_perimeter_outlet_df], ignore_index=True)
        filtered_sorted_df = group_1_df[~group_1_df['OutletID'].isin(visited_points['OutletID'])]
        filtered_sorted_df.info()
        new_map = draw_optimal_path(visited_points, new_map, G, random_color, group_feature)
  
    return visited_points, new_map

def get_html_from_map(new_map):
    """Get HTML string from folium.Map object"""
    tmpfile = BytesIO()
    new_map.save(tmpfile, close_file=False)
    html = tmpfile.getvalue().decode()
    return html

# import sys

# def get_used_libraries():
#     used_libraries = set()
#     for module_name, module in sys.modules.items():
#         if hasattr(module, '__file__'):
#             library_name = module_name.split('.')[0]  # Lấy phần tên của thư viện
#             used_libraries.add(library_name)
#     return used_libraries

# def download_csv(dataframe, filename):
#     csv = dataframe.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:text/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
#     return href

def download_excel(dataframe, filename):
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter', options={'strings_to_utf8': True}) as writer:
        dataframe.to_excel(writer, index=False, encoding='utf-8')
    excel_buffer.seek(0)
    excel_data = excel_buffer.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    return href

def main():
    st.markdown("<h1 style='text-align: center; font-size: 55px;'>Traveling Salesman Problem</h1>", unsafe_allow_html=True)

    # Upload files
    st.header("1. Upload Excel File")

    # Kiểm tra số lượng file đã tải lên
    uploaded_files = st.file_uploader("Upload Excel file", type=["xlsx"], accept_multiple_files=True)

    # # Hiển thị thông tin về file đã upload
    # if uploaded_files:
    #     st.write("Uploaded files:")
    #     for uploaded_file in uploaded_files:
    #         st.write(uploaded_file.name)
    
    dataframes = {}
    data = None
    final_df1 = None

    if uploaded_files:
        dataframes, data = process_uploaded_files(uploaded_files)

        no_oulet = st.slider("Select number outlet:", 0, 40, 30, 1)
        st.text(f"Selected number: {no_oulet}")

        # Tạo text input cho vị trí (location)
        location = st.text_input("Nhập vị trí (location):")

        # network_type
        network_type = 'bike'
        
        if location:
            st.header("2. Result")
            
            G = ox.graph_from_place(location, network_type=network_type)
            st.text("Loaded Map Done")
            
            data['Longitude'] = data['Longitude'].astype(float)
            data['Latitude'] = data['Latitude'].astype(float)
            
            cleaned_latitude = remove_outliers_iqr(data['Latitude'])
            cleaned_longitude = remove_outliers_iqr(data['Longitude'])

            # Làm sạch dữ liệu
            cleaned_data = data[(data['Latitude'].isin(cleaned_latitude)) & (data['Longitude'].isin(cleaned_longitude))]
            
            all_data, new_map = Create_square(cleaned_data, no_oulet)
            all_data = Create_RD(all_data)
            
            sovongchay = all_data['SRD'].value_counts().index[-1] + 1
            st.text('So Vong Lap: ' + str(sovongchay))     
            # st.dataframe(all_data)   
            # folium_static(new_map)
            
            visited_points_list = []
            
            for i in range(1, sovongchay):
                st.text('Dang la lan thu ' + str(i))
                print('Dang la lan thu ' + str(i))
                # Filter data for the current group (i)
                group_df = all_data[all_data['SRD'] == i]

                colors = ['black', 'lightblue', 'gray', 'blue', 'lightgreen', 'purple', 'red', 'green', 'white', 'darkblue', 'orange', 'pink', 'yellow']
                random_color = random.choice(colors)
                
                # Create visited_points, new_map, and layer_control for the current group
                visited_points_i, new_map = create_path_2(group_df, G, new_map, random_color, i)

                # Append visited_points to the list
                visited_points_list.append(visited_points_i)

                print('Chay Xong Lan thu ' + str(i))

            # Create a Layer Control
            layer_control = folium.LayerControl().add_to(new_map)

            print('Tao Danh Sach Final')
            # Khởi tạo DataFrame rỗng
            thu_danhsach = pd.DataFrame()

            # Duyệt qua từng DataFrame trong visited_points_list
            for i, df in enumerate(visited_points_list):
                # Tạo cột 'List' và gán giá trị là số thứ tự của df + 1
                df['List'] = i + 1
                # Tạo cột 'Sequence' và gán giá trị từ 1 đến chiều dài của df
                df['Sequence'] = range(1, len(df) + 1)
                # Kết hợp DataFrame hiện tại vào thu_danhsach
                thu_danhsach = pd.concat([thu_danhsach, df], ignore_index=True)
            
            print('Tao Group Cho Map')
            # In ra kết quả
            # st.dataframe(thu_danhsach)
            # folium_static(new_map, width=1000, height=800)
            # folium_static(new_map)

            # Tải new_map về dưới dạng HTML
            html = get_html_from_map(new_map)
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="map.html">Download Map</a>'
            st.markdown(href, unsafe_allow_html=True)    

            # # Tải dataframe về dưới dạng CSV
            # href_csv = download_csv(thu_danhsach, "thu_danhsach")
            # st.markdown(href_csv, unsafe_allow_html=True) 

            # Sử dụng hàm download_excel để tải DataFrame về dưới dạng Excel
            href_excel = download_excel(thu_danhsach, "thu_danhsach")
            st.markdown(href_excel, unsafe_allow_html=True)

            print('Da chay xong')
            st.markdown("<h3 style='text-align: center; font-size: 30px;'>FINISH</h1>", unsafe_allow_html=True)
            
            # import psutil
            # import os
            # # Lấy thông tin quy trình hiện tại
            # process = psutil.Process(os.getpid())

            # # In ra tổng lượng memory đã sử dụng (bằng megabyte)
            # print("Tổng lượng memory đã sử dụng:", process.memory_info().rss / 1024 ** 2, "MB")

            # # In ra lượng memory đang sử dụng (bằng megabyte)
            # print("Lượng memory đang sử dụng:", process.memory_info().vms / 1024 ** 2, "MB")
            
            # # Lấy thông tin về CPU
            # cpu_count = psutil.cpu_count(logical=False)  # Số lượng CPU vật lí
            # cpu_percent = psutil.cpu_percent(interval=1)  # Phần trăm CPU đang được sử dụng trong 1 giây

            # print("Số lượng CPU vật lí:", cpu_count)
            # print("Phần trăm CPU đang được sử dụng:", cpu_percent)

            # used_libraries = get_used_libraries()
            # print("Các thư viện được sử dụng trong quá trình chạy:")
            # print(used_libraries)      
                        
if __name__ == '__main__':
    main()        