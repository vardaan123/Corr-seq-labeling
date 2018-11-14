import shutil
import os,sys
outf=open(sys.argv[1],'w')
dirs=os.listdir('./')
print len(dirs)
count=0
for d in dirs:
	if os.path.isfile(d):
		continue
	print d
	#path=os.path.join('./',d,'performance.json')
	#path=os.path.join('./',d,'test_predictions_task0.tsv')
	path=os.path.join('./',d,sys.argv[2])
	if os.path.isfile(path):
		count+=1
		outf.write(d+'\n')
	#elif os.path.isdir(os.path.join('./',d)) and d!='jbsublogs' and d!='reports':
	#	shutil.rmtree(os.path.join('./',d))
print count
outf.close()
