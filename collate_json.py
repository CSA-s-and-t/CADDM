import json

'''ldm_dict = dict()
ldm_files = ["C:/Users/silas/Downloads/ldm.json", "C:/Users/silas/Downloads/ldm_2.json", "C:/Users/silas/Downloads/ldm_3.json", "C:/Users/silas/Downloads/ldm_4.json", "C:/Users/silas/Downloads/ldm_5.json"]
for ldm in ldm_files:
    f = open(ldm, 'r')
    ldm_dict.update(json.load(f))
    f.close()

with open('ldm.json', 'r') as f:
    json.dump(ldm_dict, f)'''

se = set()
with open('ldm.json', 'r') as f:
    dic = json.load(f)
    for i in dic.keys():
        se.add(i.split('/')[3])
print(len(se))