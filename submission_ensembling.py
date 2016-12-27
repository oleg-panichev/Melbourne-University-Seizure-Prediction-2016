import pandas as pd
import numpy as np
import matplotlib
from sklearn.isotonic import isotonic_regression

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_hist(probas, title):
    fig = plt.figure()
    plt.hist(probas, bins=50)  # plt.hist passes it's arguments to np.histogram
    plt.title(title)
    plt.xlim([0, 1])
    plt.grid()
    plt.draw()
    return fig

path = 'submits/'
figpath = path + 'fig/'
# fnames = ['submission_76867.csv', 'submission_76363.csv'] # 0.77599 - ada5 + gb1000 (0.679, 0.915)
# fnames = ['submission_75813.csv', 'submission_76363.csv', 'submission_76867.csv'] # 0.77693 - ada5 + db1000 + napoleon (0.801, 0.931)
# fnames = ['submission_lr_starter.csv', 'submission_76363.csv', 'submission_76867.csv'] # 0.75 (0.585, 0.849)
# fnames = ['submissionxgb400.csv', 'submission_76363.csv', 'submission_76867.csv'] # 0.77404 (0.780, 0.923)
# fnames = ['submission_75813.csv', 'submissionxgb400.csv', 'submission_76363.csv', 'submission_76867.csv'] # - (0.832, 0.935)
# fnames = ['submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv'] # - (0.778, 0.922)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv'] # 0.77719 (0.831, 0.934)
# fnames = ['submission_75813.csv', 'submissionxgb400.csv', 'submissionxgb800.csv', 'submission_76363.csv',
# 'submission_76867.csv'] # 0.77528 (0.856, 0.94)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_bagging100.csv', 'submission_76363.csv',
# 'submission_76867.csv'] # 0.77227 (0.825, 0.927)
fnames = ['submission_75813.csv', 'submissionxgb400.csv', 'submissionxgb800.csv', 'submission_bagging100.csv',
          'submission_76363.csv', 'submission_76867.csv'] # - (0.843, 0.932)
# fnames = ['submission_75813.csv', 'submissionxgb400.csv', 'submissionxgb800.csv', 'submission_bagging100.csv',
#           'submission_ada10_0.75577.csv', 'submission_76363.csv', 'submission_76867.csv'] # 0.77265 - (0.813, 0.916)
# fnames = ['submission_ada5_0.csv', 'submission_76867.csv'] # 0.78365 (0.807, 0.951)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv'] # 0.79778 (0.778, 0.906)
# fnames = ['submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv'] # 0.74831 (0.9998, 0.9999)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv'] # 0.80277 s(0.779, 0.9)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', 'submission_ada5_3.csv',
#           'submission_ada5_4.csv', 'submission_ada5_5.csv'] # 0.79800 (0.804, 0.907)
#           'submission_ada5_0.csv'] # 0.79778 (0.778, 0.906)
# fnames = ['submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv'] # 0.74831 (0.9998, 0.9999)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
#           'submission_ada5_starterOld_spectralV1_cv073491_lb073453.csv'] # 0.79721 (0.768, 0.893)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
#           'submission_rf_starterOld_spectralV0.csv'] # 0.80942 (0.772, 0.894)
# fnames = ['submission_75813.csv', # napoleon
#           'submissionxgb800.csv', # xgb800
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv', #gb1000
#           'submission_76867.csv',# ada5 ?
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', # ada5 simple norm
#           'submission_rf_starterOld_spectralV0.csv', # rf
#           'submission_ada66_reina_spectralV0_cv073084.csv' # ada5 reina
#           ] # 0.80747 (0.754, 0.881)
# fnames = ['submission_75813.csv', # napoleon
#           'submissionxgb800.csv', # xgb800
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv', #gb1000
#           'submission_76867.csv',# ada5 ?
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', # ada5 simple norm
#           'submission_ada5_robust.csv', # ada5 robust scale
#           'submission_rf_starterOld_spectralV0.csv', # rf
#           'submission_ada66_reina_spectralV0_cv073084.csv' # ada5 reina
#           ] # (0.754, 0.88)
# fnames = ['submission_75813.csv', # napoleon
#           'submissionxgb800.csv', # xgb800
#           'submissionxgb.csv', # xgb()
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv', #gb1000
#           # 'submission_76867.csv',# ada5 ?
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', # ada5 simple norm
#           'submission_ada5_robust.csv', # ada5 robust scale
#           'submission_rf_starterOld_spectralV0_0.csv', #'submission_rf_starterOld_spectralV0_1.csv',
#           #'submission_rf_starterOld.csv', 'submission_rf_starter.csv', # rf
#           'submissionrf10.csv', # rf reina
#           'submission_ada66_reina_spectralV0_cv073084.csv' # ada5 reina
#           ] # (0.749, 0.877)
# fnames = ['submission_75813.csv', # napoleon
#           'submissionxgb800.csv', # xgb800
#           'submissionxgb.csv', # xgb()
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv', #gb1000
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', # ada5 simple norm
#           'submission_ada5_robust.csv', # ada5 robust scale
#           'submission_rf_starterOld_spectralV0_0.csv', # rf
#           'submissionrf10.csv', # rf reina
#           'submission_ada66_reina_spectralV0_cv073084.csv' # ada5 reina
#           ] # 0.79958 (0.745, 0.875)
# fnames = ['submission_75813.csv', # napoleon
#           'submissionxgb800.csv', # xgb800
#           'submissionxgb.csv', # xgb()
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv', #gb1000
#           'submission_76867.csv',# ada5 ?
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', # ada5 simple norm
#           'submission_rf_starterOld_spectralV0_0.csv' # rf
#           ] # 0.80908 (0.769, 0.889)


# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'rf50_new.csv'] # 0.81122 (0.771401733489, 0.892200630574)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'rf50_new.csv'] # 0.81152 (0.753773738708, 0.881903201286)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_ada66_reina_spectralV0_cv073084.csv'] # 0.81010 (0.753773738708, 0.881903201286)
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv',
#           'rf50_new.csv'] # 0.81321 (0.774845101474, 0.892538373279)

# # Could have second place
# fnames = ['submission_75813.csv',  'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada55_starterOld_spectralV0_pav4.csv', #'submission_ada5_robust.csv',
#           'submission_rf_starterOld_spectralV0_0.csv','submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'submission_svmrbf_starterold_spectralv0.csv',
#           'rf50_new.csv'
#           ] # (0.77706328655, 0.895035697806)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv',
#           'submissionxgb_reina.csv',
#           'rf50_new.csv'] # 0.0.80893 (0.779385183997, 0.893593983077)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv'] # 0.81355 (0.778240137829, 0.893138222673)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv', 'gb1000_reina.csv'] # 0.80724 (0.76238043253, 0.883309569459)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv'] # 0.81134 (0.78764499735, 0.897079069164)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv', 'submission_cec0_starterOld_spectralV0.csv'] # 0.81179 (0.767621866881, 0.886660600506)

# path = 'submits_add/'
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'napoleon2_reina.csv', 'rf100_new.csv', 'submission_gb100_reina.csv', 'submission_gs_076149.csv',
#           'submission_nap2_starter.csv', 'submission_xgb100_reina.csv'] # 0.80337 (0.78581105861, 0.887025423008)

# path = 'submits_add/'
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000_76363.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf100_new.csv', 'submission_gb100_reina.csv'] # 0.80698 (0.78581105861, 0.887025423008)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.81423 (0.779261305531, 0.887521801823)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv',
#           'starter_gs_ada_dt_65_entropy.csv', 'starter_gs_ada_dt_65_gini.csv'] # 0.80078 (0.772573518703, 0.888552639691)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv',
#           'starter_2.csv', 'starter_2_2.csv', 'starter_2_3.csv', 'starter_2_5.csv'] # 0.76878 (0.752937995058, 0.876956658863)

# fnames = ['submission_75813.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.80780 ()

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv', 'starter_1.csv'] # 0.80525

# fnames = ['75813_new.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.80540

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # n=2
#
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submissionxgb800.csv','submissionxgb800.csv',
#           'submission_gb1000.csv', 'submission_gb1000.csv', 'submission_gb1000.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv']




# fnames = ['submission_AVG.csv', 'submission_AVG 2.csv'] # ------------------------------------------------------------

# 75813_new.csv
# path = 'submits_oldschool/'
# fnames = ['nap1_starter_0.csv', 'xgb800_starter_0.csv', 'gb1000_starter_0.csv',
#           'ada55_starterOld_spectralV0_0.csv',
#           'ada54_starterOld_spectralV0_0.csv',
#           'ada56_starterOld_spectralV0_0.csv',
#           'rf_starterOld_spectralV0_0.csv',
#           'rf50_reina_0.csv',
#           'gs_starterOld_spectralV0_0.csv', 'gs_starterOld_spectralV0_1.csv', 'gs_starterOld_spectralV0_2.csv'] # 0.78204 (0.781820369337, 0.895066428338)

# fnames = ['nap1_starter_0.csv', 'xgb800_starter_0.csv', 'gb1000_starter_0.csv',
#           # 'ada55_starterOld_spectralV0_0.csv', 'ada55_starterOld_spectralV0_1.csv', 'ada55_starterOld_spectralV0_2.csv',
#           # 'ada55_starterOld_spectralV0_3.csv',
#           # 'ada54_starterOld_spectralV0_0.csv', 'ada54_starterOld_spectralV0_1.csv', 'ada54_starterOld_spectralV0_2.csv',
#           # 'ada56_starterOld_spectralV0_0.csv', 'ada56_starterOld_spectralV0_1.csv', 'ada56_starterOld_spectralV0_2.csv',
#           'rf_starterOld_spectralV0_0.csv', 'rf_starterOld_spectralV0_1.csv', 'rf_starterOld_spectralV0_2.csv',
#           'rf_starterOld_spectralV0_3.csv',
#           'rf50_reina_0.csv',
#           'gs_starterOld_spectralV0_0.csv', 'gs_starterOld_spectralV0_1.csv', 'gs_starterOld_spectralV0_2.csv'] #

# path = 'submits_merge/'
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           # 'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           # 'submission_rf_starterOld_spectralV0_0.csv',
#           # 'rf50_new.csv', 'submission_gs_076149.csv',
#           # 'submission_gs_076149.csv', 'submission_gs_076149.csv',
#
#           'nap1_starter_0.csv', 'xgb800_starter_0.csv', 'gb1000_starter_0.csv',
#             # 'ada55_starterOld_spectralV0_0.csv',
#             # 'ada54_starterOld_spectralV0_0.csv',
#             # 'ada56_starterOld_spectralV0_0.csv',
#             # 'rf_starterOld_spectralV0_0.csv',
#             # 'rf50_reina_0.csv',
#             # 'gs_starterOld_spectralV0_0.csv', 'gs_starterOld_spectralV0_1.csv', 'gs_starterOld_spectralV0_2.csv'
#           ]

# fnames = ['submission_76867.csv', '../submission_starover.csv']

# path = 'submits_add/'
# fnames = ['submissionxgb800.csv', 'submission_xgb800_starter_0.csv', 'submission_xgb800_starter_oldCode.csv']

# path = 'submits/'
# fnames = ['submissionxgb.csv','submissionxgb800.csv', 'submission_xgb600.csv', 'submissionxgb400.csv',
#           'submission_xgb800_starter_ira.csv','submission_xgb800_oldNorm.csv','submission_xgb800_starter_oldCode.csv',
#           'submissionXGB800starter-1.csv','xgb800.csv']

# fnames = ['submission_gb1000.csv', 'submission_gb1000_76363.csv']
# fnames = ['submission_76867.csv','submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', 'submission_ada5_3.csv',
#           'submission_ada5_5.csv', 'submission_ada5_robust.csv','submission_ada5_oldNorm001.csv']
#
# fnames = ['submission_ada5_4.csv', 'submission_76867.csv']
# fnames = ['submission_rf_starterOld_spectralV0_0.csv','submission_rf_starterOld_spectralV0 2.csv','submission_rf_starterOld_spectralV0 3.csv']
#
# fnames = ['rf50_new.csv','rf100_new.csv', 'rf50random100.csv']
#
# fnames = ['submission_gs_076149.csv','submission_gs_starterOld_spectral_0.csv','submission_gs_starterOld_spectral_1.csv',
#           'submission_gs_starterOld_spectral_2.csv','submission_gs_starterOld_spectral_3.csv','submission_gs_starterOld_spectral_4.csv']

# fnames = ['submission_75813.csv', 'submission_nap1_starter_2.csv']

# path = 'submits_oldschool/'
# fnames = ['submission_ada5_0.csv','submission_ada5_1.csv','submission_ada5_2.csv']

# fnames = ['submission_75813.csv', 'submission_nap1_starter_2.csv',
#           'submissionxgb800.csv', 'submission_xgb800_starter_ira.csv',
#           'submission_gb1000.csv', 'submission_gb1000_76363.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', 'submission_ada5_3.csv',
#           'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv',
#           'rf50_new.csv', 'rf100_new.csv',
#           'submission_gs_076149.csv','submission_gs_starterOld_spectral_0.csv','submission_gs_starterOld_spectral_1.csv',
#           'submission_gs_starterOld_spectral_2.csv','submission_gs_starterOld_spectral_3.csv',
#           'submission_gs_starterOld_spectral_4.csv'] # 0.79898

# fnames = ['submission_nap1_starter_2.csv',
#           'submission_xgb800_starter_ira.csv',
#           'submission_gb1000_76363.csv',
#           #'submission_76867.csv',
#           'submission_ada5_1.csv', 'submission_ada5_2.csv', 'submission_ada5_3.csv',
#           'submission_rf_starterOld_spectralV0 2.csv',
#           'rf100_new.csv',
#           'submission_gs_starterOld_spectral_2.csv','submission_gs_starterOld_spectral_3.csv',
#           'submission_gs_starterOld_spectral_4.csv'] # 0.78850

# fnames = ['submission_75813.csv', 'submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
#           'submission_nap2_starter.csv', 'submission_nap2_starter.csv', 'submission_nap2_starter.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.81194 (0.794650126451, 0.894999189454)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv', 'submission_ada5_3.csv',
#           'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.81250 (0.809286318885, 0.89792041703)

# fnames = ['gb1000_reina.csv', 'rf50_new.csv',
#           'submission_75813.csv', 'submission_76867.csv', 'submission_ada5_0.csv', 'submission_ada5_1.csv',
#           'submission_ada5_2.csv', 'submission_ada5_3.csv', 'submission_ada5_4.csv', 'submission_ada5_5.csv',
#           'submission_ada5_076848.csv', 'submission_ada5_Norm without excluding x_mean and x_std separate based on train from test with probas mean bug.csv',
#           'submission_ada5_robust.csv', 'submission_ada5_starterOld_spectralV1_cv073491_lb073453.csv',
#           'submission_ada5_without excluding.csv', 'submission_ada10_0.75577.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_ada66_reina_spectralV0_cv073084.csv', 'submission_bagging100.csv', 'submission_dt5_starterOld_spectralV0.csv',
#           'submission_gb1000_76363.csv', 'submission_gb1000_starter_nle.csv', 'submission_gb1000.csv', 'submission_lr_starter.csv',
#           'submission_Norm without excluding std > 1.csv', 'submission_rf_starter 2.csv', 'submission_rf_starter.csv',
#           'submission_Norm without excluding std > 1.csv', 'submission_rf_starter 2.csv', 'submission_rf_starter.csv',
#           'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0_1.csv',
#           'submission_rf_starterOld_spectralV0_2.csv', 'submission_rf_starterOld_spectralV0.csv',
#           'submission_rf_starterOld.csv', 'submission_svmrbf_starterold_spectralv0.csv',
#           'submissionrf10.csv', 'submissionxgb.csv', 'submissionxgb400.csv', 'submissionxgb800.csv'] # (0.685883962469, 0.832945177065)
# # submissionxgb_reina.csv

# fnames = ['rf50_new.csv',
#           'submission_75813.csv', 'submission_76867.csv', 'submission_ada5_0.csv', 'submission_ada5_1.csv',
#           'submission_ada5_2.csv', 'submission_ada5_3.csv', 'submission_ada5_4.csv', 'submission_ada5_5.csv',
#           'submission_ada5_076848.csv', 'submission_ada5_Norm without excluding x_mean and x_std separate based on train from test with probas mean bug.csv',
#           'submission_ada5_robust.csv',
#           'submission_ada5_without excluding.csv', 'submission_ada10_0.75577.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_ada66_reina_spectralV0_cv073084.csv', 'submission_bagging100.csv', 'submission_dt5_starterOld_spectralV0.csv',
#           'submission_gb1000_76363.csv', 'submission_gb1000.csv',
#           'submission_rf_starter 2.csv', 'submission_rf_starter.csv',
#           'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'submission_svmrbf_starterold_spectralv0.csv',
#           'submissionxgb.csv', 'submissionxgb800.csv'] # 0.80593 (0.774015703837, 0.88404881761)

# fnames = ['rf50_new.csv', 'submission_75813.csv']

# path = 'avg/'
# fnames = ['81423.csv', '81355.csv', '81321.csv', '81250.csv']

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv'] # 0.81355 (0.778240137829, 0.893138222673)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] #
#
# # The one which works like in LB!
# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] #

# Best in private LB
fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
          'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
          'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
          'rf50_new.csv', 'submission_gs_076149.csv',
          'submission_gs_076149.csv', 'submission_gs_076149.csv', 'submission_svmrbf_starterold_spectralv0.csv'] #

# fnames = ['stack_starter.csv',
#          'stack_old.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#          'rf50_new.csv',  'submission_gs_076149.csv', 'submission_gs_076149.csv'
#           ] #(0.74607516293, 0.884356609203 )
# fnames = ['stack_starter.csv', 'stack_old.csv'] #(0.714289788307, 0.924890419006)
# fnames = ['stack_starter.csv', 'stack_old.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', 'rf50_new.csv'] # (0.763712027601, 0.899359790005 )
# fnames = ['stack_starter.csv', 'stack_old.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
# 'submission_gs_076149.csv'] # (0.781359675644, 0.907358445372)

fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
          'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
          'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
          'rf50_new.csv', 'submission_gs_076149.csv',
          'submission_gs_076149.csv', 'submission_gs_076149.csv', 'submission_svmrbf_starterold_spectralv0.csv'] #

fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv',
          'submission_ada5_0.csv', 'submission_ada5_1.csv', 'submission_ada5_2.csv',
          'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
          'rf50_new.csv', 'submission_gs_076149.csv',
          'submission_gs_076149.csv', 'submission_gs_076149.csv', 'submission_svmrbf_starterold_spectralv0.csv',
          'submission_nap2_starter.csv', 'submission_nap2_starter.csv'] #


# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] # 0.81423 (0.779261305531, 0.887521801823)

# fnames = ['submission_75813.csv', 'submissionxgb800.csv', 'submission_gb1000.csv', #'submission_76867.csv',
#           'submission_nap2_starter.csv',
#           'submission_ada5_0.csv', 'submission_ada54.csv', 'submission_ada56.csv',
#           'submission_rf_starterOld_spectralV0_0.csv', 'submission_rf_starterOld_spectralV0 2.csv', 'submission_rf_starterOld_spectralV0 3.csv',
#           'rf50_new.csv', 'submission_gs_076149.csv',
#           'submission_gs_076149.csv', 'submission_gs_076149.csv'] #


n_files = 1908
probas_buf = np.empty([n_files, len(fnames)])

for i, fname in enumerate(fnames):
    print '*** ' + fname + ' ***'
    df = pd.read_csv(path + fname, header = 0)
    data = np.array(df)

    file_names = data[:, 0]
    probas = data[:, 1]
    # print probas.shape

    # probas_n = probas
    probas_n = probas - np.min(probas)
    probas_factor = 1 / np.max(probas_n)
    probas_n = probas_n * probas_factor

    # probas_n = isotonic_regression(probas, y_min=0, y_max=1)

    probas_buf[:, i] = probas_n

    # fig = plot_hist(probas_n, fname)
    # fig.savefig(figpath + fname[:-4] + '.png', bbox_inches='tight')

c_av = 0
for i in range(0, len(fnames)):
    for j in range (i+1, len(fnames)):
        c = np.corrcoef(probas_buf[:, i], probas_buf[:, j])
        print 'Correlation between ' + fnames[i] + ' and ' + fnames[j] + ' = ' + str(c[0, 1])
        c_av += c[0, 1]
n = len(fnames)
print 'Average corrc = ' + str(c_av/float((n*n-n)/2.0))

# df = pd.DataFrame({'File': file_names, 'Class': probas_buf}, columns=['File', 'Class'], index=None)
# df.to_csv(path + 'submission_AVG.csv', sep=',', header=False, float_format='%.8f', index=False)

# print probas_buf.shape
probas_mean = np.mean(probas_buf, axis=1)

# probas_mean = np.empty(probas_buf.shape[0])
# n = 2
# for i in range(0, probas_buf.shape[0]):
#     probas_mean[i] = np.mean(np.sort(probas_buf[i])[n:-n])

probas_n = probas_mean - np.min(probas_mean)
probas_factor = 1 / np.max(probas_n)
probas_mean = probas_n * probas_factor

print '[0-0.2}: ' + str(len(probas_mean[probas_mean < 0.2]))
print '[0.2-0.4}: ' + str(len(probas_mean[(probas_mean >= 0.2) & (probas_mean < 0.4)]))
print '[0.4-0.6}: ' + str(len(probas_mean[(probas_mean >= 0.4) & (probas_mean < 0.6)]))
print '[0.6-0.8}: ' + str(len(probas_mean[(probas_n >= 0.6) & (probas_mean < 0.8)]))
print '[0.8-1}: ' + str(len(probas_mean[0.8 <= probas_mean]))

# print probas_buf
# for i in range(0, probas_buf.shape[0]):
    # print(str(i) + '. Mean = %.2f ' % np.mean(probas_buf[i]) + ' Std = %.2f ' % np.std(probas_buf[i]) +
    #       'Min = %.2f ' % np.min(probas_buf[i]) + 'Max = %.2f ' % np.max(probas_buf[i]) + 'Median = %.2f ' % np.median(probas_buf[i]))

# print probas_mean.shape
# print probas_mean

c_av = 0
for i in range(0, len(fnames)):
    c = np.corrcoef(probas_buf[:, i], probas_mean)
    print 'Correlation between ' + fnames[i] + ' and averaged = ' + str(c[0, 1])
    c_av += c[0, 1]
print 'Average corrc = ' + str(c_av/float(len(fnames)))

fname = 'AVG'
fig = plot_hist(probas_mean, fname)
# fig.savefig(figpath + fname + '.png', bbox_inches='tight')

df = pd.DataFrame({'File': file_names, 'Class': probas_mean}, columns=['File', 'Class'], index=None)
df.to_csv(path + 'submission_AVG.csv', sep=',', header=True, float_format='%.8f', index=False)

plt.show()