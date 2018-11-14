import sys,os,itertools

gold=sys.argv[1]
asr_punct_slots = sys.argv[2]
model_name = sys.argv[3].split('/')[0]

def getlabellist(file):
    f=open(file,'r')
    llist=[]
    count=0
    for line in f:
        s=line.strip().split(' ')
        s=[int(x) for x in s]
        llist.extend(s)
        count+=1
    f.close()
    return llist,count

def calculate_metrics(goldlist,predlist,num_classes):
	labellist = range(2,num_classes+1)

	S = len([i for i in range(len(goldlist)) if goldlist[i] in labellist and predlist[i] in labellist and goldlist[i]!=predlist[i]])
	C = len([i for i in range(len(goldlist)) if goldlist[i] in labellist and predlist[i] in labellist and goldlist[i]==predlist[i]])
	I = len([i for i in range(len(goldlist)) if goldlist[i] not in labellist and predlist[i] in labellist])
	D = len([i for i in range(len(goldlist)) if goldlist[i] in labellist and predlist[i] not in labellist])


	# print 'S = %d' % S
	# print 'C = %d' % C
	# print 'D = %d' % D
	# print 'I = %d' % I

	try:
		SER = (S+D+I)*1.0/(C+S+D)
		ERR = (S+D+I)*1.0/(C+S+D+I)
		return SER,ERR
	except ZeroDivisionError:
		return 0,0

def generate_metrics_task(goldlist,predlist,task_id,num_classes): # {PUNCT:[2,3],COMMA:[2],PERIOD:[3]}
	
	ser,err = calculate_metrics(goldlist,predlist,num_classes)
	print '%s,%f,%f' % (sys.argv[task_id+3],ser,err)


num_tasks = len(sys.argv)-3
#print 'num_tasks = %d' % num_tasks

f1 = open(gold,'r')

gold_labels_list = []

for i in range(num_tasks):
	gold_labels_list.append([])

gold_line_count = 0

for line_id,line in enumerate(f1):
	s=line.strip().split(' ')
	s=[int(x) for x in s]
	gold_labels_list[line_id % num_tasks].extend(s)
	gold_line_count += 1

f1.close()

pred_labels_list =[]

asr_punct_slots_list,asr_punct_slots_count = getlabellist(asr_punct_slots)

try:
	assert asr_punct_slots_count == gold_line_count/num_tasks
except:
	print 'ERROR: Line count mismatch for gold and asr slot files'
	print 'gold_line_count = %d' % gold_line_count
	print 'num tasks = %d' % num_tasks

for task_id in range(num_tasks):
	filename_task = sys.argv[task_id+3]
	predlist,predlinecount=getlabellist(filename_task)

	try:
		assert predlinecount == gold_line_count/num_tasks
	except:
		print 'ERROR: Line count mismatch for gold and pred files for task %d' % task_id
		print 'predlinecount = %d' % predlinecount
		print 'gold_line_count = %d' % gold_line_count
	pred_labels_list.append(predlist)

for task_id in range(num_tasks):
    predlist = pred_labels_list[task_id]
    num_classes = max(gold_labels_list[task_id])

    gold_labels_list_filt = [gold_labels_list[task_id][i] for i in range(len(asr_punct_slots_list)) if asr_punct_slots_list[i]==1]
    predlist_filt = [predlist[i] for i in range(len(asr_punct_slots_list)) if asr_punct_slots_list[i]==1]

    generate_metrics_task(gold_labels_list_filt,predlist_filt,task_id,num_classes)
	






