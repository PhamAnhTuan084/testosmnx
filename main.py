import osmnx as ox
import streamlit as st
from streamlit_folium import folium_static
import folium

# Thiết lập địa điểm và network_type
location = "Tỉnh Bến Tre"
network_type = 'bike'

# Sử dụng osmnx để tải dữ liệu đồng thời xây dựng biểu đồ đường đi
G = ox.graph_from_place(location, network_type=network_type)

st.text("load thành công")