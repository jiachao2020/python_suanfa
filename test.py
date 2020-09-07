def cftHireFunc(arr):
    # write code
    # 判断最小值，最小值大于0则为最大值+1，否则为1
    arr = list(arr)
    arr=sorted(arr)
    tmp=list(range(arr[0],arr[-1]+1))
    for i in arr:
        if i in tmp:
           tmp.remove(i)
    if len(tmp)==0:
        if arr[-1]<=0:
            return 1
        else: return arr[-1]+1
    elif tmp[0] <= 0:
        return 1
    else:
        return tmp[0]
print(cftHireFunc([-1,3,2,3,3243,4,5,12,7]))
