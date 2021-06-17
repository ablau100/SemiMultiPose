import torch
import numpy as np
def matmul(U, V):
	U_tensor = torch.Tensor(U.reshape((-1, U.shape[1]))).cuda()
	V_tensor = torch.Tensor(V).cuda()
	output = U_tensor.mm(V_tensor)
	prod = output.cpu().detach().numpy().astype("double")
	return prod

A = np.random.randint(1,20,25)
B = np.random.randint(1,20,35)

A = A.reshape((5,5))
B = B.reshape((5,7))

matmul(A,B)



