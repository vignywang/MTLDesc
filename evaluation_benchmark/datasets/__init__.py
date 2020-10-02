def get_dataset(name):
    mod = __import__('datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def get_sub_dataset(name):
    mod = __import__('datasets.{}'.format(name.split('_')[0]), fromlist=[_module_to_class(name)])
    attr = getattr(mod, _module_to_class(name))
    return attr


def _module_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
