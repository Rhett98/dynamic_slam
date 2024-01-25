import numpy as np

pose_path = "/home/yu/Resp/dataset/school_dataset/school_1658.kitti"
output_path = "./06.txt"
with open(pose_path, 'r') as input_file, open(output_path, 'w') as output_file:
    lines = input_file.readlines()
    # 每十行保留一行
    for i, line in enumerate(lines):
        if i % 10 == 0:
            output_file.write(line)