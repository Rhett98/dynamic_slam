output_path = "/home/yu/Resp/dataset/school_dataset/sequences/04/times.txt"
total_step = 1146
count = 0
with open(output_path, 'w') as output_file:
    next_time = float(0.000000e+00)
    for j in range(total_step):
        output_file.write(str(next_time)+"\n")
        next_time += 0.1