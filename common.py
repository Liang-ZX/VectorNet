import datetime

def cur_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
