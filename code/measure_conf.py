import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(m, h)
    return m, m-h, m+h

all_seeds_train1 = list()
all_seeds_test1 = list()
all_seeds_train2 = list()
all_seeds_test2 = list()
all_seeds_train3 = list()
all_seeds_test3 = list()
all_seeds_train4 = list()
all_seeds_test4 = list()

best_seeds_test1 = list()
best_seeds_test2 = list()
best_seeds_test3 = list()
best_seeds_test4 = list()

num_model = 30

for i in range(num_model):
    train_acc1 = list()
    test_acc1 = list()
    train_acc2 = list()
    test_acc2= list()
    train_acc3 = list()
    test_acc3 = list()
    train_acc4 = list()
    test_acc4= list()

    f1 = open('../logs/Comp1/log0%02d.out'%i, mode='r')
    f2 = open('../logs/Comp2/log0%02d.out'%i, mode='r')
    f3 = open('../logs/Comp3/log0%02d.out'%i, mode='r')
    f4 = open('../logs/a1_normal/log0%02d.out'%i, mode='r')

    for i in range(50):
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        line4 = f4.readline()

    for i in range(50, 150):
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        line4 = f4.readline()

        if(line1 and line2 and line3 and line4):
            train_acc1.append(np.array(line1.split('    ')[2], dtype=np.float32))
            test_acc1.append(np.array(line1.split('    ')[4], dtype=np.float32))

            train_acc2.append(np.array(line2.split('    ')[2], dtype=np.float32))
            test_acc2.append(np.array(line2.split('    ')[4], dtype=np.float32))
            
            train_acc3.append(np.array(line3.split('    ')[2], dtype=np.float32))
            test_acc3.append(np.array(line3.split('    ')[4], dtype=np.float32))

            train_acc4.append(np.array(line4.split('    ')[2], dtype=np.float32))
            test_acc4.append(np.array(line4.split('    ')[4], dtype=np.float32))
            
            if(np.array(line1.split('    ')[0], dtype=np.int32)==149):
                best_seeds_test1.append(np.array(line1.split('    ')[5], dtype=np.float32))
                best_seeds_test2.append(np.array(line2.split('    ')[5], dtype=np.float32))
                best_seeds_test3.append(np.array(line3.split('    ')[5], dtype=np.float32))
                best_seeds_test4.append(np.array(line4.split('    ')[5], dtype=np.float32))

        else:
            break

    f1.close()
    f2.close()
    f3.close()
    f4.close()


    all_seeds_train1.append(np.array(train_acc1))
    all_seeds_test1.append(np.array(test_acc1))
    all_seeds_train2.append(np.array(train_acc2))
    all_seeds_test2.append(np.array(test_acc2))
    all_seeds_train3.append(np.array(train_acc3))
    all_seeds_test3.append(np.array(test_acc3))
    all_seeds_train4.append(np.array(train_acc4))
    all_seeds_test4.append(np.array(test_acc4))

# 1 : C1 2: C2, 3: C3, 4: M5
all_seeds_test1 = np.array(all_seeds_test1)
all_seeds_test2 = np.array(all_seeds_test2)
all_seeds_test3 = np.array(all_seeds_test3)
all_seeds_test4 = np.array(all_seeds_test4)

best_seeds_test1 = np.array(best_seeds_test1)
best_seeds_test2 = np.array(best_seeds_test2)
best_seeds_test3 = np.array(best_seeds_test3)
best_seeds_test4 = np.array(best_seeds_test4)

minimum = np.min(all_seeds_test4, axis=0)
average = np.mean(all_seeds_test4, axis=0)
maximum = np.max(all_seeds_test4, axis=0)

#mean_confidence_interval(best_seeds_test1)
#mean_confidence_interval(best_seeds_test2)
#mean_confidence_interval(best_seeds_test3)
#mean_confidence_interval(best_seeds_test4)


