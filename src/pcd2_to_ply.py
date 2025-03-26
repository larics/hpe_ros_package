
#!/usr/bin/python3

import rospy
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from sensor_msgs.msg import PointCloud2

PCL_TOPIC = "/camera/depth_registered/points"

class Pcd2Ply:
    def __init__(self):
        rospy.init_node('pcd2ply_node', anonymous=True)
        self.subscriber = rospy.Subscriber(
            PCL_TOPIC,  # Change this to your actual topic name
            PointCloud2,
            self.cpcl_cb
        )
        rospy.loginfo_once("pcd2ply_node has started")

    def pcl_cb(self, ros_point_cloud):
            rospy.loginfo("Received point cloud")
            xyz = np.array([[0,0,0]])
            rgb = np.array([[0,0,0]])
            #self.lock.acquire()
            gen = pc2.read_points(ros_point_cloud, skip_nans=True)
            int_data = list(gen)

            for x in int_data:
                test = x[3] 
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f' ,test)
                i = struct.unpack('>l',s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000)>> 16
                g = (pack & 0x0000FF00)>> 8
                b = (pack & 0x000000FF)
                # prints r,g,b values in the 0-255 range
                            # x,y,z can be retrieved from the x[0],x[1],x[2]
                xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
                rgb = np.append(rgb,[[r,g,b]], axis = 0)

            out_pcd = o3d.geometry.PointCloud()    
            out_pcd.points = o3d.utility.Vector3dVector(xyz)
            out_pcd.colors = o3d.utility.Vector3dVector(rgb)
            rospy.loginfo("Saving pointcloud")
            o3d.io.write_point_cloud("/root/test_cloud.ply",out_pcd)
            rospy.loginfo("Pointcloud saved")
            exit()   

    def _pcl_cb(self, ros_point_cloud):
        # Convert PointCloud2 to a NumPy array
        points_list = []
        for point in pc2.read_points(ros_point_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        # Convert to Open3D Point Cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_list))

        # Save to .ply file
        o3d.io.write_point_cloud("/root/output.ply", pcd)
        rospy.loginfo("Saved point cloud to output.ply")

    def cpcl_cb(self, ros_point_cloud):
        # Convert PointCloud2 to a NumPy array
        points_list = []
        colors_list = []
        
        for point in pc2.read_points(ros_point_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            # Extract points (x, y, z)
            points_list.append([point[0], point[1], point[2]])
            
            # Extract color (RGB packed as a single float)
            rgb = point[3]
            rgb = struct.unpack('I', struct.pack('f', rgb))[0]
            
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF
            # Normalize the colors to [0, 1]
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

        # Convert to Open3D Point Cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_list))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors_list))

        # Save to .ply file with colors
        o3d.io.write_point_cloud("/root/output.ply", pcd)
        rospy.loginfo("Saved point cloud with colors to output.ply")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        pcd2plyNode = Pcd2Ply()
        pcd2plyNode.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down pcd2ply_node")
