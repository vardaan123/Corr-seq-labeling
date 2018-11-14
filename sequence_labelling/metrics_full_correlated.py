import sys,os,itertools

gold=sys.argv[1]
model_name = sys.argv[2].split('/')[0]

def getlabellist(file):
    f=open(file,'r')
    llist=[]
    count=0
    for line in f:
        s=line.strip().split(' ')
        s=[int(x) for x in s]
        llist.extend(s)
        count+=1
    return llist,count

def calculate_metrics(goldlist,predlist,positive_token_list):
    positives = 0
    predicted_positives = 0

    for token in positive_token_list:
        positives += goldlist.count(token)
        predicted_positives += predlist.count(token)

    tpr=0.0
    for i in range(len(goldlist)):
        if (goldlist[i] in positive_token_list) and (predlist[i] in positive_token_list):
            tpr+=1
    try:
        prec = tpr/predicted_positives
        recall = tpr/positives
        f1score = 2 * prec * recall / (prec+recall)
        return prec, recall, f1score
    except ZeroDivisionError:
        return 0,0,0

def calculate_SER(goldlist,predlist,num_classes):
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

def calculate_overall_metrics(goldlist,predlist,labellist):
    expected_positives = 0
    predicted_positives = 0
    correct_positives = 0.0

    for token in labellist:
        expected_positives += goldlist.count(token)
        predicted_positives += predlist.count(token)

    for i in range(len(goldlist)):
        if (goldlist[i] in labellist) and (predlist[i] in labellist) and goldlist[i]==predlist[i]:
            correct_positives += 1
    try:
        prec = correct_positives/predicted_positives
        recall = correct_positives/expected_positives
        f1score = 2 * prec * recall / (prec+recall)
        return prec, recall, f1score
    except ZeroDivisionError:
        return 0,0,0

def generate_metrics_task(goldlist,predlist,punct_list,task_id): # {PUNCT:[2,3],COMMA:[2],PERIOD:[3]}
    num_classes = max(goldlist)
    labellist = range(2,num_classes+1)
    SER, ERR = calculate_SER(goldlist,predlist,num_classes)

    for punct in punct_list:
        prec, recall, f1score = calculate_metrics(goldlist,predlist,punct)
        print '%s,%s,%f,%f,%f,%f,%f' % (sys.argv[task_id+2],':'.join([str(x) for x in punct]),prec,recall,f1score,SER,ERR)

    prec_overall,recall_overall,f1score_overall = calculate_overall_metrics(goldlist,predlist,labellist)
    print '%s,%s,%f,%f,%f,%f,%f' % (sys.argv[task_id+2],'overall',prec_overall,recall_overall,f1score_overall,SER,ERR)



num_tasks = len(sys.argv)-2
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

pred_labels_list =[]

for task_id in range(num_tasks):
    filename_task = sys.argv[task_id+2]
    predlist,predlinecount=getlabellist(filename_task)

    try:
        assert predlinecount == gold_line_count/num_tasks
    except:
        print 'ERROR: Line count mismatch for gold and pred files for task %d' % task_id
        print 'predlinecount = %d' % predlinecount
        print 'gold_line_count = %d' % gold_line_count
    pred_labels_list.append(predlist)
    
def fn(l):
    totallist = []
    for i,j in itertools.product(range(len(l)),range(len(l))):
        if i<=j:
            totallist.append(l[i:j+1])
    return totallist


for task_id in range(num_tasks):
    predlist = pred_labels_list[task_id]
    num_classes = max(gold_labels_list[task_id])
    
    labellist = range(2,num_classes+1)
    punct_list = fn(labellist)

    generate_metrics_task(gold_labels_list[task_id],predlist,punct_list,task_id)
    






