from math import sqrt
from joblib import Parallel, delayed
import time

def long_task(i:int) -> int:
	time.sleep(1)
	return i

if __name__ == "__main__":
	# st = time.time()
	# #ss = [sqrt(i**2) for i in range(100000)]
	# ss = [long_task(i**2) for i in range(10)]
	# print(time.time() - st)
	# print(type(ss))
	# #print(ss)
	print('----------')
	st = time.time()
	#aa = Parallel(n_jobs=2, backend="threading")(delayed(sqrt)(i**2) for i in range(100000))
	aa = Parallel(n_jobs=10, backend="threading")(delayed(long_task)(i**2) for i in range(10))
	print(type(aa))
	print(aa)
	print(time.time() - st)
	#print(aa)
