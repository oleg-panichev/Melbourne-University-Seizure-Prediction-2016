import pandas as pd
import numpy as np

def ensemble(probas_buf, coef):
	p = np.zeros(probas_buf.shape[0])
	n = sum(coef)
	for i in range(probas_buf.shape[0]):
		for j in range(probas_buf.shape[1]):
			p[i] += probas_buf[i, j] * coef[j]
		p[i] /= float(n)

	return p

model_names = ['gb', 'xgb', 'vot', 'ada55_0', 'ada55_1', 'ada55_2', 'rf_0', 'rf_1', 'rf_2', 'gs', 'rfreina']
model_coef = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1]

# model_names = ['gb', 'xgb', 'vot', 'ada55_0', 'ada55_1', 'ada55_2', 'rf_0', 'gs', 'rfreina']
# model_coef = [1, 1, 1, 1, 1, 1, 1, 3, 1]

# model_names = ['gb', 'xgb', 'vot', 'ada55_0', 'rf_0', 'gs', 'rf_reina']
# model_coef = [1, 1, 1, 3, 3, 3, 1]

# model_names = ['vot', 'rf_0', 'gs', 'rfreina']
# model_coef = [1, 3, 3, 1]

spath = 'submissions/'

n_files = 1908
probas_buf = np.empty([n_files, len(model_names)])

for i, model_name in enumerate(model_names):
    print '*** ' + model_name + ' ***'
    df = pd.read_csv(spath + 'submission_' + model_name + '.csv', header = 0)
    data = np.array(df)

    file_names = data[:, 0]
    probas = data[:, 1]

    # probas_n = probas - np.min(probas)
    # probas_factor = 1 / np.max(probas_n)
    # probas_n = probas_n * probas_factor

    probas_buf[:, i] = probas

# probas_mean = np.mean(probas_buf, axis=1)
probas_mean = ensemble(probas_buf, model_coef)

probas_n = probas_mean - np.min(probas_mean)
probas_factor = 1 / np.max(probas_n)
probas_mean = probas_n * probas_factor

df = pd.DataFrame({'File': file_names, 'Class': probas_mean}, columns=['File', 'Class'], index=None)
df.to_csv(spath + 'submission_ensemble.csv', sep=',', header=True, float_format='%.8f', index=False)
