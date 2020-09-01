#
# Created by ZhangYuyang on 2020/8/31
#


def get_labeler(name):
    names = name.split('.')
    f_name = '.'.join(names[:-1])
    c_name = names[-1]
    mod = __import__('{}.{}'.format(__name__, f_name), fromlist=[''])
    return getattr(mod, c_name)



