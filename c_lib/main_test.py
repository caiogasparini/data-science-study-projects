import time
import pandas as pd
from preprocessor import Preprocessor
from memory_optimizer import DataFrameOptimizer

f_path = r'C:\Users\caio_\OneDrive\Documentos\Caio\PROJETOS - CIÊNCIAS DE DADOS\Git e GitHub\data-science-study-projects\unsupervised-learning\clustering\customer-clustering\datas\segmentation data.csv'

#################### Preprocessor Class test ####################
# df = pd.read_csv(f_path)

# pp: Preprocessor = Preprocessor()
# dif_amount_v5 = pp.feature_engeneering(df['Amount'], df['V5'])
#################################################################

################# DataFrameOptimizer Class test #################
dfo = DataFrameOptimizer()

## df = dfo.read_csv('polars', f_path)
## df_b = dfo.read_csv('pandas', f_path)
## print(df.describe())
## print(df_b.describe())

antes = time.time()
df = dfo.pl_read_csv(f_path)
depois = time.time()
print(df.describe())
print(depois-antes)
#################################################################