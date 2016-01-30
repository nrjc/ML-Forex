from fann2 import libfann
import re
f = open('testingdata', 'r')
a=[]
for line in f:
    b=re.sub('\n','',line).split('\t')
    c=[]
    for thing in b:
		c.append(float(thing))
    a.append(c)
	
	
ann = libfann.neural_net()
ann.create_from_file("new.net")
totalnumber=0.0
numberright=0.0
for test in a:
	output=test.pop()
	if(ann.run(test)>=0.5):
		predicted=1;
	else:
		predicted=0;
	totalnumber+=1
	if (output==predicted):
		numberright+=1
	
print numberright, totalnumber
print numberright/totalnumber

