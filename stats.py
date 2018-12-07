def precision(A, c):
	tp = A[c][c]
	fp = sum(A[:,c]) - A[c][c]
	if tp+fp == 0:
		return 0
	return tp/(tp + fp)

def recall(A, c):
	tp = A[c][c]
	fn = sum(A[c]) - A[c][c]
	if tp + fn == 0:
		return 0
	return tp/(tp + fn)

def F1(A, c):
	p = precision(A,c)
	r = recall(A,c)
	if p+r == 0:
		return 0
	return (2*p*r)/(p+r)

def F1overall(A):
	f = 0
	for c in range(len(A)):
		f += F1(A,c)
	return f/len(A)