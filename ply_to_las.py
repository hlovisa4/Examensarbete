# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:40:17 2026

@author: Lovisa
"""

from plyfile import PlyData
import numpy as np
import laspy
import matplotlib.pyplot as plt

ply_path = r"C:\Users\digit\Downloads\731160_7134030fixedname_noisysegments.ply"
out_las  = r"C:\Users\digit\Downloads\731160_7134030fixedname_noisysegments.las"

ply = PlyData.read(ply_path)
v = ply["vertex"]

# --- Required geometry ---
x = np.asarray(v["x"], dtype=np.float64)
y = np.asarray(v["y"], dtype=np.float64)
z = np.asarray(v["z"], dtype=np.float64)

# --- Your DL outputs ---
semantic = np.asarray(v["semantic_pred"])
instance = np.asarray(v["instance_pred"])
instance_sc = np.asarray(v["instance_score"])


# Pick safe dtypes (adjust if you know the exact ranges)
# semantic labels usually fit in uint16; instance ids often need uint32/int32
semantic = semantic.astype(np.uint32, copy=False)
instance = instance.astype(np.uint32, copy=False)
instance_sc = instance_sc.astype(np.uint32, copy=False)

# Create LAS header
hdr = laspy.LasHeader(point_format=3, version="1.2")

# Use offsets/scales to preserve precision
hdr.x_scale = 0.001
hdr.y_scale = 0.001
hdr.z_scale = 0.001
hdr.x_offset = float(np.floor(x.min()))
hdr.y_offset = float(np.floor(y.min()))
hdr.z_offset = float(np.floor(z.min()))

las = laspy.LasData(hdr)
las.x = x
las.y = y
las.z = z

# Add extra dimensions for predictions
las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.uint32))
las.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.uint32))
las.add_extra_dim(laspy.ExtraBytesParams(name="instance_score", type=np.uint32))

las["semantic_pred"] = semantic
las["instance_pred"] = instance
las["instance_score"] = instance_sc

las.write(out_las)
print("Wrote:", out_las)
print("LAS dims:", las.point_format.dimension_names)



las2 = laspy.read(out_las)
print(np.unique(las2["semantic_pred"])[:20])
print(len(np.unique(las2["instance_pred"])))
print(len(np.unique(las2["instance_score"][las2["instance_score"] < 200])))

mask = las2["instance_score"] < 200
print(len(np.unique(las2["instance_pred"][mask])))

