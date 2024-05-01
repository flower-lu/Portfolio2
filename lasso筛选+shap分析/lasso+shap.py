import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.impute import SimpleImputer
import baostock as bs
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from  MyTT import *
import warnings
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

# 1. 数据收集
lg = bs.login()
rs = bs.query_history_k_data_plus("sh.600000",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2013-01-01', end_date='2023-01-01',
    frequency="d", adjustflag="3")
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.extend(rs.get_row_data())

# 将数据转换为DataFrame
data = np.array(data_list)
n_rows = len(data) // 14
stock_data = pd.DataFrame(data.reshape(n_rows, 14), columns=rs.fields)

# 将open,high,low,close,preclose,volume,amount,turn,pctChg转换为数值型
for col in ['open','high','low','close','preclose','volume','amount','turn','pctChg']:
    stock_data[col] = stock_data[col].replace('', np.nan).astype(float)

# 计算股票收益率
stock_data['return'] = stock_data['close'].pct_change().shift(-1)
stock_data = stock_data.dropna()

date = stock_data['date']
CLOSE=stock_data['close'];  OPEN=stock_data['open'];   HIGH=stock_data['high'];
LOW=stock_data['low']; VOL=stock_data['volume'] #基础数据定义

MA5=MA(CLOSE,5)
MA10=MA(CLOSE,10)
MA20=MA(CLOSE,20)
DIF,DEA,MACD=MACD(CLOSE)
K,D,J=KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3)
RSI=RSI(CLOSE, N=24)
BIAS1,BIAS2,BIAS3=BIAS(CLOSE, L1=6, L2=12, L3=24)
BOLL_UPPER, BOLL_MID, BOLL_LOWER=BOLL(CLOSE, N=20, P=2)
PSY,PSYMA=PSY(CLOSE, N=12, M=6)
CCI=CCI(CLOSE, HIGH, LOW, N=14)
ATR=ATR(CLOSE, HIGH, LOW, N=20)
BBI=BBI(CLOSE, M1=3, M2=6, M3=12, M4=20)
PDI, MDI, ADX, ADXR=DMI(CLOSE, HIGH, LOW, M1=14, M2=6)
TAQ_UP, TAQ_MID, TAQ_DOWN=TAQ(HIGH, LOW, N=10)
KTN_UPPER, KTN_MID, KTN_LOWER=KTN(CLOSE, HIGH, LOW, N=20, M=10)
TRIX, TRMA=TRIX(CLOSE, M1=12, M2=20)
EMV, MAEMV = EMV(HIGH, LOW, VOL, N=14, M=9)
DPO, MADPO= DPO(CLOSE, M1=20, M2=10, M3=6)
AR, BR = BRAR(OPEN, CLOSE, HIGH, LOW, M1=26)
MTM, MTMMA = MTM(CLOSE, N=12, M=6)
MASS, MA_MASS = MASS(HIGH, LOW, N1=9, N2=25, M=6)
ROC, MAROC = ROC(CLOSE, N=12, M=6)
EMA1, EMA2 = EXPMA(CLOSE, N1=12, N2=50)
OBV = OBV(CLOSE, VOL)
MFI = MFI(CLOSE, HIGH, LOW, VOL, N=14)  # MFI指标是成交量的RSI指标
ASI, ASIT = ASI(OPEN, CLOSE, HIGH, LOW, M1=26, M2=10)



# 2. 特征工程
# 计算技术指标和基本面指标作为特征
stock_data['MA5'] = MA5
stock_data['MA10'] = MA10
stock_data['MA20'] = MA20
stock_data['DIF'] = DIF
stock_data['DEA'] = DEA
stock_data['MACD'] = MACD
stock_data['K'] = K
stock_data['D'] = D
stock_data['J'] = J
stock_data['RSI'] = RSI
stock_data['BIAS1'] = BIAS1
stock_data['BIAS2'] = BIAS2
stock_data['BIAS3'] = BIAS3
stock_data['BOLL_UPPER'] = BOLL_UPPER
stock_data['BOLL_MID'] = BOLL_MID
stock_data['BOLL_LOWER'] = BOLL_LOWER
stock_data['PSY'] = PSY
stock_data['PSYMA'] = PSYMA
stock_data['CCI'] = CCI
stock_data['ATR'] = ATR
stock_data['BBI'] = BBI
stock_data['PDI'] = PDI
stock_data['MDI'] = MDI
stock_data['ADX'] = ADX
stock_data['ADXR'] = ADXR
stock_data['TAQ_UP'] = TAQ_UP
stock_data['TAQ_MID'] = TAQ_MID
stock_data['TAQ_DOWN'] = TAQ_DOWN
stock_data['KTN_UPPER'] = KTN_UPPER
stock_data['KTN_MID'] = KTN_MID
stock_data['KTN_LOWER'] = KTN_LOWER
stock_data['TRIX'] = TRIX
stock_data['TRMA'] = TRMA
stock_data['EMV'] = EMV
stock_data['MAEMV'] = MAEMV
stock_data['DPO'] = DPO
stock_data['MADPO'] = MADPO
stock_data['AR'] = AR
stock_data['BR'] = BR
stock_data['MTM'] = MTM
stock_data['MTMMA'] = MTMMA
stock_data['MASS'] = MASS
stock_data['MA_MASS'] = MA_MASS
stock_data['ROC'] = ROC
stock_data['MAROC'] = MAROC
stock_data['EMA1'] = EMA1
stock_data['EMA2'] = EMA2
stock_data['OBV'] = OBV
stock_data['MFI'] = MFI
stock_data['ASI'] = ASI
stock_data['ASIT'] = ASIT

# 添加更多的技术指标和基本面指标,总共50个指标


# 3. 特征选择
# 使用LASSO回归进行特征选择

# 分离特征和目标变量
X = stock_data.drop(['return', 'code', 'date'], axis=1)
y = stock_data['return']

# 填充X中的NaN值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# # LASSO特征选择
# lasso = LassoCV(cv=5, random_state=42, alphas=[0.001, 0.01, 0.1, 1, 10, 100])
# lasso.fit(X, y)
# feature_mask = SelectFromModel(lasso, prefit=True)
# X_selected = feature_mask.transform(X)


# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 假设 X_scaled 是一个 numpy 数组，我们首先将其转换为 DataFrame
X_scaled = pd.DataFrame(X_scaled)
# 生成示例数据和噪声
data = np.random.randint(0, 100, X_scaled.shape[1])  # 生成与 X_scaled 列数相同数量的随机整数
noise = np.random.randint(-10, 11, X_scaled.shape[0] * X_scaled.shape[1]).reshape(X_scaled.shape)
# 将噪声添加到原始数据中
noisy_data = X_scaled.values + noise

# 定义四种滤波函数
def amplitude_filter(data, threshold=10):
    filtered_data = np.clip(data, data.mean() - threshold, data.mean() + threshold)
    return filtered_data

def median_filter(data, window_size=5):
    # 计算窗口的起始和结束索引，确保它们是整数
    start = (window_size - 1) // 2
    end = start + window_size
    # 初始化一个空的数组来存储滤波后的数据
    filtered_data = np.empty_like(data)
    # 对每一列进行中值滤波
    for i in range(data.shape[1]):
        # 对每一行的窗口内的数据计算中值
        filtered_data[:, i] = np.median(data[:, i][start:end], axis=0)
    return filtered_data
def mean_filter(data, window_size=5):
    weights = np.ones(window_size) / window_size
    filtered_data = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='same'), axis=0, arr=data)
    return filtered_data
def recursive_average_filter(data, alpha=0.3):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    return filtered_data
# 应用滤波
filtered_amplitude = amplitude_filter(noisy_data, threshold=10)
filtered_median = median_filter(noisy_data)
filtered_mean = mean_filter(noisy_data)
filtered_recursive = recursive_average_filter(noisy_data)

# 将滤波后的数据转换为 DataFrame
X_filtered_amplitude = pd.DataFrame(filtered_amplitude, columns=X_scaled.columns, index=X_scaled.index)
X_filtered_median = pd.DataFrame(filtered_median, columns=X_scaled.columns, index=X_scaled.index)
X_scaled = pd.DataFrame(filtered_mean, columns=X_scaled.columns, index=X_scaled.index)
X_filtered_recursive = pd.DataFrame(filtered_recursive, columns=X_scaled.columns, index=X_scaled.index)

# 现在你有多个增强后的数据集，可以用于进一步的分析或模型训练

# 使用 LassoCV 进行交叉验证并选择最佳的 alpha 值
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
# 获取最佳 alpha 值
best_alpha = lasso.alpha_
# 打印最佳 alpha 值
print(f"Best alpha: {best_alpha}")
# 使用最佳 alpha 值创建 Lasso 模型
lasso_best = Lasso(alpha=best_alpha, random_state=42)
# 拟合最佳 alpha 值的 Lasso 模型
lasso_best.fit(X_scaled, y)

# 使用 SelectFromModel 选择特征
feature_mask = SelectFromModel(lasso_best, prefit=True)
X_selected = feature_mask.transform(X_scaled)


# 手动创建特征名称列表
feature_names = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn',
                 'pctChg', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD','K', 'D', 'J',
                 'RSI', 'BIAS1', 'BIAS2', 'BIAS3', 'BOLL_UPPER', 'BOLL_MID', 'BOLL_LOWER',
                 'PSY', 'PSYMA', 'CCI', 'ATR', 'BBI', 'PDI', 'MDI', 'ADX', 'ADXR', 'TAQ_UP',
                 'TAQ_MID', 'TAQ_DOWN', 'KTN_UPPER', 'KTN_MID', 'KTN_LOWER', 'TRIX', 'TRMA',
                 'MAEMV','EMV','DPO', 'MADPO', 'AR', 'BR', 'MTM', 'MTMMA', 'MASS', 'MA_MASS',
                 'ROC', 'MAROC', 'EMA1', 'EMA2', 'OBV', 'MFI', 'ASI', 'ASIT',
                 ]

selected_feature_indices = np.flatnonzero(feature_mask.get_support())  # 获取选择的特征索引
# selected_features = [feature_names[i] for i in selected_feature_indices]  # 根据索引获取选择的特征名称
# print(f"Selected features: {', '.join(selected_features)}")

# 确保 selected_feature_indices 中的每个索引都是有效的
if all(i < len(feature_names) for i in selected_feature_indices):
    # 获取选择的特征名称
    selected_features = [feature_names[i] for i in selected_feature_indices]
    print(f"Selected features: {', '.join(selected_features)}")
else:
    print("Error: Some selected feature indices are out of range.")

# 如果没有选择任何特征,手动选择一些特征
if len(selected_features) == 0:
    print("No features were selected. Using default features...")
    selected_features = ['open', 'high', 'low', 'close', 'volume']  # 手动选择一些特征
    X_selected = X[:, [feature_names.index(f) for f in selected_features]]
else:
    X_selected = X[:, selected_feature_indices]  # 根据索引提取选择的特征

# 检查X_selected和y的形状是否一致
if X_selected.shape[0] != y.shape[0]:
    print(f"Warning: X_selected and y have different lengths ({X_selected.shape[0]} vs {y.shape[0]})")
    min_length = min(X_selected.shape[0], y.shape[0])
    X_selected = X_selected[:min_length]
    y = y[:min_length]

# 划分训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.4, random_state=1,shuffle=False,)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1,shuffle=False,)

# 4. 模型训练
# 定义模型列表
models = [
    XGBRegressor(random_state=42),
    LGBMRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    AdaBoostRegressor(random_state=42),
    MLPRegressor(random_state=42),
    SVR(),
    KNeighborsRegressor()
]
# 这几个模型如果调参，可能结果会好很多

# 模型评估
model_scores = {}
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    model_scores[model_name] = {'MSE_train': mse_train, 'MSE_val': mse_val, 'MSE_test': mse_test,
                                 'R2_train': r2_train, 'R2_val': r2_val, 'R2_test': r2_test}
    print(f"{model_name} MSE_train: {mse_train:.4f}, R2_train: {r2_train:.4f}")
    print(f"{model_name} MSE_val: {mse_val:.4f}, R2_val: {r2_val:.4f}")
    print(f"{model_name} MSE_test: {mse_test:.4f}, R2_test: {r2_test:.4f}")

    # 输出每个模型的预测值与真实值对比的图
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)  # 2行2列的子图，这是左上角的子图
    plt.plot(y_train.index, y_train, label='True Values (Train)')
    plt.plot(y_train.index, y_pred_train, label='Predicted Values (Train)')
    plt.title('Train Predictions vs True Values')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(y_val.index, y_val, label='True Values (Val)')
    plt.plot(y_val.index, y_pred_val, label='Predicted Values (Val)')
    plt.title('Val Predictions vs True Values')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(y_test.index, y_test, label='True Values (Test)')
    plt.plot(y_test.index, y_pred_test, label='Predicted Values (Test)')
    plt.title('Test Predictions vs True Values')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=300)
    # plt.show()
    # plt.close()

# 选择最佳模型
best_model_name = max(model_scores, key=lambda x: model_scores[x]['R2_val'])
best_model = [m for m in models if m.__class__.__name__ == best_model_name][0]
print(f"Best model: {best_model_name}")


# 5. 模型优化
# 使用随机搜索优化超参数
if isinstance(best_model, RandomForestRegressor):
    param_distributions = {
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300, 400],
        'max_features': ['sqrt', 'log2']
    }
elif isinstance(best_model, (XGBRegressor, LGBMRegressor)):
    param_distributions = {
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
else:
    # 对于其他模型,可以设置不同的超参数空间
    param_distributions = {}

random_search = RandomizedSearchCV(best_model, param_distributions=param_distributions,
                                   n_iter=10, cv=5, scoring='r2', random_state=42)
random_search.fit(X_train, y_train)

# 6. 模型评估
# 五折交叉验证
cv_scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='r2')
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")

# 7. 模型可解释性
print('重复交叉验证')
np.random.seed(1)  # Reproducibility
CV_repeats = 10
random_states = np.random.randint(10000, size=CV_repeats)  # 制作一个长度 = CV_repeats 的 0 到 10000 之间的随机整数列表，作为不同的数据分割

X = pd.DataFrame(X_selected)
y = pd.DataFrame(y)

df = pd.concat([X, y], axis=1)

# 使用dict跟踪每次 CV 重复时每个观测值的 SHAP 值
shap_values_per_cv = dict()
for sample in X.index:
    # Create keys for each sample
    shap_values_per_cv[sample] = {}
    # Then, keys for each CV fold within each sample
    for CV_repeat in range(CV_repeats):
        shap_values_per_cv[sample][CV_repeat] = {}
        # 内层字典的结构是 [样本索引][交叉验证重复次数][折叠编号]。

for i, CV_repeat in enumerate(range(CV_repeats)): #-#-#
    # Verbose
    print('\n------------ CV Repeat number:', CV_repeat)
    # Establish CV scheme
    CV = KFold(n_splits=5, shuffle=True, random_state=random_states[i]) # Set random state

    ix_training, ix_test = [], []
    # 循环遍历每个折叠并将训练和测试索引附加到上面的列表中
    for train_index, test_index in CV.split(X):  # 注意这里应该是 X，而不是 df
        ix_training.append(train_index)
        ix_test.append(test_index)

    # 循环遍历每个外层折叠并提取 SHAP 值
    for train_outer_ix, test_outer_ix in zip(ix_training, ix_test):
        # Verbose
        print('\n------ Fold Number:', len(ix_training) - i)  # i 从 0 开始，所以用 len(ix_training) - i 来计数

        # 使用正确的索引来分割 X 和 y
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        model = RandomForestRegressor(random_state=10) # 随机状态的可重复性（每次都有相同的结果）
        fit = model.fit(X_train, y_train)
        yhat = fit.predict(X_test)
        result = mean_squared_error(y_test, yhat)
        print('RMSE:', round(np.sqrt(result), 4))

        # 使用SHAP来解释预测
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 提取每个样本每个折叠的 SHAP 信息
        for i, test_index in enumerate(test_outer_ix):
            shap_values_per_cv[test_index][CV_repeat] = shap_values[i] #-#-#

# 将每个样本每个交叉验证重复的SHAP值平均为一个值以进行绘制
# 建立列表来保存平均 Shap 值、它们的标准以及它们的最小值和最大值
average_shap_values, stds, ranges = [], [], []

for i in range(0, len(X)):
    df_per_obs = pd.DataFrame.from_dict(shap_values_per_cv[i])  # 获取样本号 i 的所有 SHAP 值
    # 获取每个样本的相关统计数据
    average_shap_values.append(df_per_obs.mean(axis=1).values)
    stds.append(df_per_obs.std(axis=1).values)
    ranges.append(df_per_obs.max(axis=1).values-df_per_obs.min(axis=1).values)

shap.summary_plot(np.array(average_shap_values), X, show = False)
plt.title('Average SHAP values after 10x cross-validation')
plt.savefig('Average SHAP-1.png')

# 由于我们的结果已经在多次重复的交叉验证中进行了平均，因此它们比仅执行一次的简单训练/测试拆分更稳健且可信。
ranges = pd.DataFrame(ranges) ; ranges.columns = X.columns
import seaborn as sns; from matplotlib import pyplot as plt
# Transpose dataframe to long form
values, labels = [], []
for i in range(len(ranges.columns)):
    for j in range(len(ranges)):
        values.append(ranges.T[j][i])
        labels.append(ranges.columns[i])
long_df = pd.DataFrame([values,labels]).T ; long_df.columns = ['Values', 'Features']


title = 'Range of SHAP values per sample across all\ncross-validation repeats'
xlab, ylab = 'SHAP Value Variability', 'SHAP range per sample'
sns.catplot(data = long_df, x = 'Features', y = 'Values').set(xlabel = xlab, ylabel = ylab,
                                                                            title = title)
plt.xticks(rotation=45)
plt.savefig('Average SHAP-2.png')
# plt.show()

mean_abs_effects = long_df.groupby(['Features']).mean()

standardized = long_df.groupby(long_df.Features).transform(lambda x: x/x.mean()) ; standardized['Features'] = long_df.Features

title = 'Scaled Range of SHAP values per sample \nacross all cross-validation repeats'
sns.catplot(data = standardized, x = 'Features', y = 'Values').set(xlabel = 'SHAP Value Variability Scaled by Mean',
                                                                            title = title)
plt.xticks(rotation=45)
plt.savefig('Average SHAP-3.png')
# plt.show()
