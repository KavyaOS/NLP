import os
import datetime

file_num = 0
file = open('transcripts_with_word_time_offsets/6.txt','r')
for line in file:
    file_num = file_num + 1
    words =  line.split()
    start_time = words[3]
    start_time = start_time[:-1]

    end_time = words[5]
    command = "ffmpeg " + "-i " + "input_videos/6.mp4 " + "-ss " + str(datetime.timedelta(seconds=float(start_time))) + " -to " + str(datetime.timedelta(seconds=float(end_time))) + " -c:v " + 'libx264 ' + 'trimmed_videos/6.mp4/6_'+ str(file_num) +'.mp4 '
    os.system(command)