import sys,os
gold=sys.argv[1]
pred=sys.argv[2]

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

goldlist,goldlinecount=getlabellist(gold)
predlist,predlinecount=getlabellist(pred)
num_classes = max(predlist)
#print len(goldlist), len(predlist)
if goldlinecount!=predlinecount:
    print "ERROR"
    print gold,pred

#positive -label 2
if num_classes == 3:
    type = 'COMMA'
    # print '-------------------COMMA--------------------'
else:
    type = 'PUNCT'
    # print '-------------------PUNCTUATION--------------------'
positives=goldlist.count(2)
predicted_positives=predlist.count(2)
tpr=0.0
for i in range(len(goldlist)):
    if goldlist[i]==2 and predlist[i]==2:
        tpr+=1
prec = tpr/predicted_positives
recall = tpr/positives
f1score = 2 * prec * recall / (prec+recall)
print '%s,%s,%s,%f,%f,%f' % (gold,pred,type,prec,recall,f1score)
# print 'Precision : %f' % (prec)
# print 'Recall : %f' % (recall)
# print 'F1 Score : %f' % (f1score)

if num_classes == 2:
    sys.exit()

#positive - label 3
# print '-------------------PERIOD--------------------'
positives=goldlist.count(3)
predicted_positives=predlist.count(3)
tpr=0.0
for i in range(len(goldlist)):
    if goldlist[i]==3 and predlist[i]==3:
        tpr+=1
type = 'PERIOD'
prec = tpr/predicted_positives
recall = tpr/positives
f1score = 2 * prec * recall / (prec+recall)
print '%s,%s,%s,%f,%f,%f' % (gold,pred,type,prec,recall,f1score)
# print 'Precision : %f' % (prec)
# print 'Recall : %f' % (recall)
# print 'F1 Score : %f' % (f1score)


#positive - label 2 & 3
# print '-------------------PUNCTUATION--------------------'
positives=goldlist.count(2)+goldlist.count(3)
predicted_positives=predlist.count(2)+predlist.count(3)
tpr=0.0
for i in range(len(goldlist)):
    if goldlist[i]!=1 and predlist[i]!=1:
        tpr+=1
type = 'PUNCT'
prec = tpr/predicted_positives
recall = tpr/positives
f1score = 2 * prec * recall / (prec+recall)
print '%s,%s,%s,%f,%f,%f' % (gold,pred,type,prec,recall,f1score)
# print 'Precision : %f' % (prec)
# print 'Recall : %f' % (recall)
# print 'F1 Score : %f' % (f1score)



