import open3d as o3d
import pcl


point_cloud = pcl.PointCloud()  # 你的点云数据
poses = [...]  # 你的位姿数据

o3d_point_cloud = o3d.geometry.PointCloud()
o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.to_array())


o3d.visualization.draw_geometries([o3d_point_cloud])


downpcd = o3d_point_cloud.voxel_down_sample(voxel_size=0.05)


global_map = o3d.geometry.PointCloud()
for pose in poses:
    transformed_pcd = downpcd.transform(pose)
    global_map += transformed_pcd


o3d.visualization.draw_geometries([global_map])
o3d.io.write_point_cloud("global_map.ply", global_map)