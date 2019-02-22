def averagePrecision(groudtruth, score):
    num = len(groudtruth)
    t = sum(groudtruth)
    res = []
    for i in range(num):
        res.append([score[i], groudtruth[i], i])
    res.sort(key = lambda x:x[0], reverse=True)
    cnt = 0
    ap = 0
    for i, v in enumerate(res):
        if v[1] == 1:
            cnt += 1
            ap += cnt / (i + 1)
    return ap / t

def mAP(res):
    aps = []
    for v in res:
        groudtruth = v['gt']
        score = v['score']
        aps.append(averagePrecision(groudtruth, score))
    return aps