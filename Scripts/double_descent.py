from multiprocessing import Pool
from models import LUPTS, linearRegression
from data import linear_gaussian_system
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

SYSTEMS = 5
SAMPLES = 100
SYS_FEATURES = range(20, 200, 10)
SYS_OUT_FEATURES = 10
SEQ_LEN = 4
SEQ_STEP = 1
WORKERS = 8

MODELS = [
    LUPTS(),
    linearRegression()
]


def fit_predict_score(*args):
    sys, seed = args
    (train_x, train_y), (test_x, test_y) = sys.get_train_test_data(SEQ_LEN, SEQ_STEP, 0.5, SAMPLES, seed)
    d = sys.no_features
    scores = []
    for m in MODELS:
        m.fit(train_x, train_y)
        y_hat = m.predict(test_x)
        score = r2_score(test_y[:, 0], y_hat[:, 0])
        scores.append((m.name, d, float(score)))
    return scores


if __name__ == '__main__':

    configs = []
    for feats in SYS_FEATURES:
        for _ in range(SYSTEMS):
            sys = linear_gaussian_system(feats, feats, test_size=1000, seed=np.random.randint(0, 2 ** 16))
            configs.append((sys, np.random.randint(0, 2 ** 16)))

    pool = Pool(WORKERS)
    print('Starting pool of workers... ')
    results = pool.starmap(fit_predict_score, configs)
    sorted_res = {}
    for res in results:
        for r in res:
            key = (r[0], r[1])
            if key not in sorted_res:
                sorted_res[key] = [r[2]]
            else:
                sorted_res[key].append(r[2])

    fig = plt.figure(figsize=(5, 3))
    fontsize = 21
    ols_means = np.array([np.mean(sorted_res['OLS', d]) for d in SYS_FEATURES])
    ols_std = np.array([np.std(sorted_res['OLS', d]) for d in SYS_FEATURES])
    plt.fill_between(SYS_FEATURES, ols_means - ols_std, ols_means + ols_std, alpha=0.1, color='#2E4053')
    plt.plot(SYS_FEATURES, ols_means, 'o-', color='#273746', label='OLS', linewidth=3)

    lupts_means = np.array([np.mean(sorted_res['LuPTS (Linear)', d]) for d in SYS_FEATURES])
    lupts_std = np.array([np.std(sorted_res['LuPTS (Linear)', d]) for d in SYS_FEATURES])
    plt.fill_between(SYS_FEATURES, lupts_means - lupts_std, lupts_means + lupts_std, alpha=0.1, color='#D35400')
    plt.plot(SYS_FEATURES, lupts_means, 'o--', color='#D35400', label='LuPTS', linewidth=3)
    plt.legend(fontsize=17)
    plt.xlabel(r'Features, d', fontsize=fontsize)
    plt.ylabel('R²', fontsize=fontsize)
    ax = plt.gca()
    ax.set_yticks(np.arange(-1.1, 1.1, 0.1))
    ax.set_yticks(np.arange(-1.1, 1.1, 0.05), minor=True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(which='minor', zorder=1, alpha=0.1)
    plt.grid(which='major', zorder=2, alpha=0.2)
    plt.ylim(0.3, 0.9)
    plt.tight_layout()
    plt.savefig('double_descent_r2.pdf', format='pdf', facecolor='#ffffff')
    plt.show()
