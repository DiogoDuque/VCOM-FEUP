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
hits_detailed = {"arrabida": 0, "camara": 0, "clerigos": 0, "musica": 0, "serralves": 0}
misses_detailed = {"arrabida": {"count": 0, "mistaken": []},
                    "camara": {"count": 0, "mistaken": []},
                    "clerigos": {"count": 0, "mistaken": []},
                    "musica": {"count": 0, "mistaken": []},
                    "serralves": {"count": 0, "mistaken": []}}
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
                hits_detailed[expectedLabel] += 1
            else:
                misses += 1
                misses_detailed[expectedLabel]["count"] += 1
                misses_detailed[expectedLabel]["mistaken"].append(pName)
            break

print("[GENERAL] Hits: %d(%.2f%%), Undetected: %d(%.2f%%), Misses: %d(%.2f%%), Imgs processed: %d" %
(hits, 100*hits/len(testImgs), undetected, 100*undetected/len(testImgs), misses, 100*misses/len(testImgs), len(testImgs)))

for monument in ["arrabida", "camara", "clerigos", "musica", "serralves"]:
    m_hits = hits_detailed[monument]
    m_misses = misses_detailed[monument]["count"]
    m_total = m_hits+m_misses
    if m_total == 0:
        continue
    m_misses_mistaken = {}
    if m_misses > 0:
        for mistake in misses_detailed[monument]["mistaken"]:
            if mistake not in m_misses_mistaken:
                m_misses_mistaken[mistake] = 1
            else:
                m_misses_mistaken[mistake] += 1
    m_hits_perc = (100*m_hits)/m_total
    m_misses_perc = (100*m_misses)/m_total
    m_misses_mistaken = str(m_misses_mistaken)
    print("[%s] Hits: %d(%.2f%%), Misses: %d(%.2f%%) %s, Imgs amount: %d" % (monument, m_hits, m_hits_perc, m_misses, m_misses_perc, m_misses_mistaken, m_total))
