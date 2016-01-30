from fann2 import libfann
import re
f = open('EURUSDTESTINGDATA', 'r')
a=[]
for line in f:
    b=re.sub('\n','',line).split('\t')
    c=[]
    for thing in b:
		c.append(float(thing))
    a.append(c)
	
	
ann = libfann.neural_net()
ann.create_from_file("new.net")
for test in a:
	output=test.pop()
	print "output",output
	print "test values",test
	predicted=ann.run(test)
	print "predicted",predicted
	error=output-predicted[0]
	print "error",error

