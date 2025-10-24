import numpy as np

#Params to interperpolate
start = [0,0.0,14.0]
end = [0, 95.0, 7.75]
num_pts = 20.0
d_line = ((np.array(end)-np.array(start)))/num_pts
#current point starts at the starting point 
curr_pt = start
#File to write
file_json = open("/media/uyoo/MyPassport/POE/Mesh/cable1.json","w")
for i in range(int(num_pts)):
    added = ""
    last = ","
    if i==0:
        added = "["
    file_json.write( added + "[" + str(curr_pt[0]) + "," +  str(curr_pt[1]) + "," +  str(curr_pt[2]) + "], \n")
    #update the current point
    curr_pt = curr_pt+d_line
file_json.write( "[" + str(curr_pt[0]) + "," +  str(curr_pt[1]) + "," +  str(curr_pt[2]) + "]]")
