import osmnx as ox
import streamlit as st
from streamlit_folium import folium_static
import folium

# Thiết lập địa điểm và network_type
location = "Tỉnh Bến Tre"
network_type = 'bike'

# Sử dụng osmnx để tải dữ liệu đồng thời xây dựng biểu đồ đường đi
G = ox.graph_from_place(location, network_type=network_type)

# Tạo một bản đồ Folium
m = folium.Map(location=[10.2270, 106.6914], zoom_start=12)  # Điều chỉnh tọa độ và zoom tùy theo vùng bạn đang quan tâm

# Thêm dữ liệu biểu đồ vào bản đồ Folium
ox.plot_graph_folium(G, graph_map=m, popup_attribute='name', edge_width=2)

# Hiển thị bản đồ Folium trên Streamlit
folium_static(m)