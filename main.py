import osmnx as ox
import streamlit as st
from streamlit_folium import folium_static
import folium

# Thiết lập địa điểm và network_type
location = "Thủ Đức VietNam"
network_type = 'drive'

# Sử dụng osmnx để tải dữ liệu đồng thời xây dựng biểu đồ đường đi
G = ox.graph_from_place(location, network_type=network_type)

st.text("load thành công")