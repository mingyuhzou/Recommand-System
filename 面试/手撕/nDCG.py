import numpy as np

y    = np.array([1,0,1,0,0])
pred = np.array([1,1,0,0,0])

def nDCG(y, y_pred, k=10):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    k = min(k, len(y))

    order = np.argsort(y_pred)[::-1]
    rel = y[order][:k]

    # gain
    gain = 2**rel - 1

    discount = np.log2(np.arange(2, k + 2))
    DCG = np.sum(gain / discount) # 注意求和

    # ideal DCG
    ideal_rel = np.sort(y)[::-1][:k]
    ideal_gain = 2**ideal_rel - 1
    iDCG = np.sum(ideal_gain / discount)

    if iDCG == 0:
        return 0.0
    return DCG / iDCG

#
print(nDCG(y, pred))