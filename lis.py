import os
path = '/mnt/data/docking.log'
with open(path, 'r') as fp:
    obj = fp.readlines()

remain_list = []
t_wv = []
for i, each in enumerate(obj):
    if 't_wv' in each:
        c = ''
        c += '{} {} {}\n'.format(float(obj[i + 1][:-1]), float(obj[i + 2][:-1]), float(obj[i + 3][:-1]))
        t_wv.append(c)
for each in obj:
    if 'z y x' in each:
        a = each.split(":")[-1][:-1]
        b = a.split(" ")
        c = ''
        for num in b:
            try:
                x = float(num)
                c += '{} '.format(x)
            except:
                pass
        c += '\n'
        remain_list.append(c)
with open('/mnt/data/zyx.txt', 'w') as fp:
    fp.writelines(remain_list)
with open('/mnt/data/t_wv.txt', 'w') as fp:
    fp.writelines(t_wv)
print('')