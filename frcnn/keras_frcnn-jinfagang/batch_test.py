import os
import pickle

from predict_kitti import predict

class Dummy:
    path = ""

# get classes
with open('config.pickle', 'rb') as f_in:
    cfg = pickle.load(f_in)
cfg.use_horizontal_flips = False
cfg.use_vertical_flips = False
cfg.rot_90 = False

class_mapping = cfg.class_mapping
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

# get test dataset
f = open("kitti_simple_label_test.txt")
fStr = f.read()
fLines = fStr.split("\n")
if(fLines[-1]==""):
    del fLines[-1]
testImgs = [[fLine.split(",")[0], fLine.split(",")[-1]] for fLine in fLines]

# test!
d = Dummy()
d.path = [ti[0] for ti in testImgs]
results = predict(d)

#print stuff
#print(results)
hits = 0
misses = 0
undetected = 0
for i in range(0, len(testImgs)):
    expectedLabel = testImgs[i][1]
    result = results[i]
    if not result:
        undetected += 1
        continue
    classId = list(result.keys())[0]
    for classPair in class_mapping.items():
        pName, pId = classPair
        if classId == pId:
            if pName == expectedLabel:
                hits += 1
            else:
                misses += 1
            break

print("Hits: %d(%.2f%%), Undetected: %d(%.2f%%), Misses: %d(%.2f%%), Imgs processed: %d" %
(hits, 100*hits/len(testImgs), undetected, 100*undetected/len(testImgs), misses, 100*misses/len(testImgs), len(testImgs)))

