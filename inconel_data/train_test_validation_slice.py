import numpy as np
from sklearn.utils import shuffle




inputs_ = np.load('./input_test.npy')
outputs_ = np.load('./output_test.npy')

input_data, output_data = shuffle(inputs_, outputs_)
#condition = (input_data[:, -1] >= 15) & (input_data[:, -1] <= 28)
#indices = np.where(condition)[0]

#input_data = input_data[condition]

#output_data = output_data[indices]



#print (np.min(inputs_[:,0]))



n = 8500

input_data_train_val = input_data[:n,:]
output_data_train_val = output_data[:n,:]


test_inputs = input_data[n:, :]
test_outputs = output_data[n:, :]

print (len(input_data_train_val))
print (len(test_inputs))


np.save('input_train_data.npy', input_data_train_val)
np.save('output_train_data.npy', output_data_train_val)

np.save('input_test_data.npy', test_inputs)
np.save('output_test_data.npy', test_outputs)


