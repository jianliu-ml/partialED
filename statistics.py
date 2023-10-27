import json

num_sen = 0
num_tok = 0

filename = 'research/SLasMRC/data/EE/include_times_and_values/json/train.json'
for line in open(filename):
    json_dic = json.loads(line)
    num_sen += len(json_dic['sentences'])
    for sen in json_dic['sentences']:
        num_tok += len(sen)


filename = 'research/SLasMRC/data/EE/include_times_and_values/json/dev.json'
for line in open(filename):
    json_dic = json.loads(line)
    num_sen += len(json_dic['sentences'])
    for sen in json_dic['sentences']:
        num_tok += len(sen)


filename = 'research/SLasMRC/data/EE/include_times_and_values/json/test.json'
for line in open(filename):
    json_dic = json.loads(line)
    num_sen += len(json_dic['sentences'])
    for sen in json_dic['sentences']:
        num_tok += len(sen)

print(num_sen)
print(num_tok)