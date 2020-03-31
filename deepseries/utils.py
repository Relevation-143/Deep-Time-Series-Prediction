# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com


class HyperParameters(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value
