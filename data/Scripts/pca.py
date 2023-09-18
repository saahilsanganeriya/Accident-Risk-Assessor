import numpy as np
import pandas as pd
import csv
np.random.seed(42)


data = [] #assume centered data
K = 5 #number of features that we want to reduce to
retained_variance = .99 #variance that we retain/want to retain
file_path = "./data/Accident Data/normalized_numerical_data.csv"


#read in data
df = pd.read_csv(file_path)

if df.shape[1] == 28:
    last_column = df.iloc[:, -1].to_numpy().reshape((-1, 1))
    df = df.iloc[:, :-1]
print(df.count)
data = df.to_numpy()
# print(data[0])
# print(data.shape)


#calculate u, s, vt. fit
temp = np.sum(data, axis=0)
temp = temp / len(data)
data = data - temp #subtract mean
U, S, Vt = np.linalg.svd(data, full_matrices=False)

#calculate wanted array using K principal axes. transform
# wanted_U = U[:, :K]
# s_array = np.diag(S[:K])
# transformed_data = np.dot(wanted_U, s_array) #this is the final product


#if we don't care about a predefined K and want to just keep a predefined variance. transform_rv
squared_S = np.square(S)
total = np.sum(squared_S)
dim = len(S)

temp_sum = 0
for i in range(len(S)):
    temp_sum += squared_S[i]
    if (temp_sum / total) >= retained_variance:
        dim = i + 1
        break
    
wanted_U = U[:, :dim]
s_array = np.diag(S[:dim])
transformed_data = np.dot(wanted_U, s_array) #same as above, should prob make into function. this is the final product
print(transformed_data)
# print(transformed_data.shape) #make sure the dimensions are correct
# print(dim) #to see how many principal axes we have with our defined retained_variance
# print(transformed_data[0])

#output data to csv file 
output_file = 'output.csv'
with open(output_file, 'w', newline='') as csv_out_file:
    writer = csv.writer(csv_out_file)
    header = []
    for i in range(1, 1 + dim):
        header.append("PCA " + str(i))
    header.append("Accident Binary")
    writer.writerow(header)
    transformed_data = np.concatenate((transformed_data, last_column), axis=1)
    for row in transformed_data:
        writer.writerow(row)