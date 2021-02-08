# first line: 1
@mem.cache
def get_data():
    data = load_svmlight_file("./rcv1_train.binary.bz2")
    return data[0], data[1]
