import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
u_cols=['user_id','age','sex','occupation','zip_code']
users=pd.read_csv('ml-100k/u.user',sep='|',names=u_cols,encoding='latin-1')

r_cols=['user_id','movie_id','rating','unix_timestamp']
ratings=pd.read_csv('ml-100k/u.data',sep='\t',names=r_cols,encoding='latin-1')

i_cols=['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
items=pd.read_csv('ml-100k/u.item',sep='|',names=i_cols,encoding='latin-1')

print(users.shape)#看看表格的行列数
users.head()#看前五行 但是没有print python运行不会显示 可以改成print(user.head())

print(ratings.shape)
ratings.head()

print(items.shape)
items.head()

r_cols=['user_id','movie_id','rating','unix_timestamp']
ratings_train=pd.read_csv('ml-100k/ua.base',sep='\t',names=r_cols,encoding='latin-1')
ratings_test=pd.read_csv('ml-100k/ua.test',sep='\t',names=r_cols,encoding='latin-1')
ratings_train.shape,ratings_test.shape

#n_users=ratings_train.user_id.unique().shape[0] #.user_id取出ratings中user_id一列 .unique()进行去重 .shape 数组的形状 返回元组(943,) .shape[0]取元组第0号元素 得到数字943 即为总用户数量
#n_items=ratings_train.movie_id.unique().shape[0] #同理 计算电影数量
n_users=ratings_train.user_id.unique().max()
n_items=ratings_train.movie_id.unique().max()

#开始创建用户电影矩阵 可用于计算用户与电影之间的相似性
data_matrix=np.zeros((n_users,n_items)) #创建二维矩阵 n_users行 n_items列   data_matrix[i,j]`：代表用户 i 给电影 j 打的评分，没有评分默认为 0
for line in ratings_train.itertuples():    #itertuples()用来遍历 DataFrame 表格的每一行，把每一行封装成元组 line 比如说第一次读到第0行 就通过line[1]（用户id） line[2]（电影id）记录用户对电影的评分line[3]
    data_matrix[line[1]-1,line[2]-1]=line[3] # Q:此处为什么要-1?   A:因为用户id和电影id是从1开始算的 但是矩阵data_matrix下标从0开始 因此需要偏移保证空间

from sklearn.metrics.pairwise import pairwise_distances #计算余弦相似度
user_similarity=1-pairwise_distances(data_matrix,metric='cosine')#计算样本集合中，每两个样本之间的距离，输出距离矩阵 data_matrix.shape (943, 1682) 行：用户；列：电影  data_matrix.T.shape (1682, 943) 行：电影；列：用户
item_similarity=1-pairwise_distances(data_matrix.T,metric='cosine') #第一个是样本矩阵 metric='cosine'：距离度量方式 cosine是余弦距离 即1-余弦值  `.T是 numpy矩阵转置  每一行作为一个向量 两两计算输出距离矩阵

def predict(ratings,similarity,type='user'):  #ratings--用户对电影评分表 similarity--余弦相似度
    rated_mask=(ratings!=0)  #【修正新增】布尔矩阵：标记"真正打过分的格子"，因为矩阵里 0 表示"没打分"而非真的打了0分
    if type=='user':#用户协同过滤
        mean_user_rating=ratings.sum(axis=1)/np.maximum(rated_mask.sum(axis=1),1)   #【修正】原为 ratings.mean(axis=1)，会把"没打分=0"的格子也算进分母，平均值被稀释到约1/6；改成只对"打过分"的电影求平均
        # .mean()对数组取平均值  axis=None 全部元素算一个平均值 axis=0 按列求平均值 axis=1 按行求平价均值    （上面这行是我的原注释，代码已按修正改掉，注释保留）
        #mean_user_rating:  每个用户的平均评分
        ratings_diff=(ratings-mean_user_rating[:, np.newaxis])*rated_mask  #【修正】末尾 *rated_mask 把"没打分"的格子置0，防止(0-均值)变成负垃圾值污染求和
        #Q:为什么要计算评分偏差?
        #A:因为有些人打分因为习惯会出现普遍高和普遍低的情况 如果直接拿原始评分去加权预测，会被用户自身打分松紧的偏差干扰结果 因此计算评分偏差矩阵来计算预测偏差
        #ratings_diff:评分偏差矩阵 = 用户原始评分 − 用户自己平均分  ratings (943,1682)是二维矩阵 而原mean_user_rating (943,)是一维数组
        #因此需要[:, np.newaxis]在列的方向新增一个维度  升维之后：(943,)->(943, 1)，变成二维列向量 :`= 取全部；np.newaxis = 插入一个大小为 1 的新维度。
        #[:, np.newaxis]：在行方向保留全部，新增列维度；一维转 (M,1) 列向量。 同理 [np.newaxis,:] 保留列方向 新增行维度 一维数组(M,)转为(1,M)
        # np.newaxis本身不改变数字大小 只改变数组形状 给计算搭建好维度
        pred=mean_user_rating[:, np.newaxis]+similarity.dot(ratings_diff)/np.maximum(np.abs(similarity).dot(rated_mask.astype(float)),1)  #【修正】分母改为"只对对方用户真实评过这部电影"的相似度求和，并防除0
        #similarity.dot(ratings_diff) 矩阵similarity 点乘 矩阵ratings_diff  可以得到二维数组 用户i对电影j的预测评分偏差之和 行是用户 列是电影
        # 矩阵点乘矩阵 比如矩阵A点乘矩阵B 要求A的列数必须等于B的行数 A的每一行和B的每一列按顺序两两相乘然后相加求和
        #矩阵A点乘向量V(二维) 要求矩阵A的列数等于V的元素个数 因此矩阵A点乘矩阵B相当于A点乘从B中按列按序取出列向量
        #【以下 np.array([np.abs(similarity).sum(axis=1)]).T 是原分母写法，已被上面【修正】的分母替换，注释保留】
        #np.abs(similarity).sum(axis=1) abs(similarity):计算余弦相似度绝对值 .sum(axis=1)按行求和相加 得到一维数组余弦绝对值之和
        #np.array([ sum_row ]) 比如一维数组[1,2,3] 写成[[1,2,3]]就相当于把原来的一维数组作为新二维数组的第一行 (3,)->(1,3)
        #np.array()把 Python 的列表 (list)、元组，转换成 numpy 的数组（ndarray），这样就可以做广播、mean、dot 矩阵运算。
        #.T把二维行向量转换而二维列向量 实现匹配广播维度，实现矩阵按行除法归一化
        #最后用每个用户平均评分加上预测评分偏差得到预测评分
    elif type =='item':#物品协同过滤
        pred =ratings.dot(similarity)/np.maximum(rated_mask.astype(float).dot(np.abs(similarity)),1)  #【修正】分母改为"只对用户真实评过分的电影"求和并防除0
        #ratings矩阵点乘similarity矩阵 对于ratings每一行就是用户i对电影1,2,3,---的评分 对于similarity每一列就是电影j对电影1,2,3---的相似度 相乘累加就是用户i对电影j的评分加权求和
        #分子得到每一行代表用户，每一列代表电影；分母得到二维行向量，列数等于电影数量。这里是 numpy 广播除法，尾部维度对齐，依靠广播实现每一列全部用户预测值除以该电影的相似度权重总和
    return pred
user_prediction=predict(data_matrix,user_similarity,type='user')
item_prediction=predict(data_matrix,item_similarity,type='item')

test_matrix=np.zeros((n_users,n_items))
for line in ratings_test.itertuples():
    test_matrix[line[1]-1,line[2]-1]=line[3]  #【修正】原来写成 test_matrix((...)) 用了圆括号，是"调用函数"；取元素要用方括号 [ , ]

#==================== 评估：在测试集上算 RMSE（越小越好）====================
def rmse(prediction,test):
    #测试集每一行是(user_id,movie_id,真实评分)，用 id-1 当矩阵下标取出对应的预测值
    preds=prediction[test['user_id'].values-1, test['movie_id'].values-1] 
    actual=test['rating'].values  
    return np.sqrt(np.mean((preds-actual)**2))

print('\n===== RMSE 评估（越小越好）=====')
base=ratings_train.rating.mean()
print('基线(全预测全局均值 %.2f): RMSE = %.4f' % (base, np.sqrt(np.mean((base-ratings_test.rating.values)**2))))
print('用户协同过滤: RMSE = %.4f' % rmse(user_prediction,ratings_test))
print('物品协同过滤: RMSE = %.4f' % rmse(item_prediction,ratings_test))
