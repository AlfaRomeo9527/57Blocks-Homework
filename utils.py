import csv
import time
import math


def show_data(file_path ):
    with open(file_path,"r") as f:
        reader=csv.reader(f)
        rows_data=list(reader)
        print(rows_data[:5])


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
