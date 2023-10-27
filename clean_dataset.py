import pickle

def read_addition_dataset(filename):
    res = []
    for line in open(filename):
        fields = line.strip().split(' ')
        doc_id, sen_id, w_id, w = fields[0].split('|')
        t = ''
        for x in fields[2:]:
            if x != 'O':
                t = x[2:]
                break
        if t:
            res.append([doc_id, sen_id, w_id, t])
    return res


def add_data(test_data, test_data_add):
    for elem in test_data_add:
        doc_id, sen_id, w_id, t = elem
        sen_id, w_id = int(sen_id), int(w_id)
        
        print(test_data[sen_id][1][w_id], t)
        temp = [[w_id, w_id, t]]
        test_data[sen_id][-1].append(temp)


if __name__ == '__main__':
    test_data_add = read_addition_dataset('data/test.txt')
    dev_data_add = read_addition_dataset('data/dev.txt')

    with open('data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    train, dev, test = data['train'], data['val'], data['test']
    
    add_data(test, test_data_add)
    add_data(dev, dev_data_add)

    with open('data/data_add.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)