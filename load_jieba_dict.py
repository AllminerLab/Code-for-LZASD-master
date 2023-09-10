import os

'''
with open('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/jieba_fast/dict.txt', 'a') as jieba_dict:
    # 载入自有词库
    
    for dict_file in os.listdir('/Users/lizhifm/code/python/导流广告/dataset/词库/'):
        if dict_file == '.DS_Store':
            continue
        with open('/Users/lizhifm/code/python/导流广告/dataset/词库/'+dict_file) as f:
            for line in f.readlines():
                line = line.replace('\n','').replace('\ufeff','').replace(' ','')
                if line == '':
                    continue
                line = line + ' 3 n\n'
                jieba_dict.write(line)    
'''
with open('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/jieba_fast/dict.txt', 'w') as jieba_dict:
    # 载入自有词库
    with open('/Users/lizhifm/code/python/导流广告/dataset/词库/自建词库.txt') as f:
        for line in f.readlines():
            line = line.replace('\n','').replace('\ufeff','').replace(' ','')
            if line == '':
                continue
            line = line + ' 3 n\n'
            jieba_dict.write(line)   
# 注意把jieba.cache删除，不然不会重新生成词库缓存
print('加载完成')
