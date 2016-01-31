from fann2 import libfann
import re
nameoffile="jy"
f = open(nameoffile+"testing", 'r')
a=[]
for line in f:
    b=re.sub('\n','',line).split('\t')
    c=[]
    for thing in b:
		c.append(float(thing))
    a.append(c)
	
	
ann = libfann.neural_net()
ann.create_from_file(nameoffile+".net")
totalnumber=0.0
numberright=0.0
predictup=0
for test in a:
	output=test.pop()
	result=ann.run(test)
	print test
	print result
	if(result[0]>=0.5):
		predicted=1
		predictup+=1
	else:
		predicted=0
	print "We predict ",predicted
	print "The output was ",output
	totalnumber+=1
	if (output==predicted):
		numberright+=1
	
print "Total number of right guesses:",numberright,"Total number of trades:", totalnumber
print "percentage correct:",numberright/totalnumber
print "Total buy", predictup

