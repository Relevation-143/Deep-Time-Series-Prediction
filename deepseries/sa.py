# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 16:08
"""
from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('my_experiment')
ex.observers.append(MongoObserver())

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.capture
def some_function(a, foo, bar=10):
    print(a, foo, bar)

@ex.main
def my_main():
    some_function(1, 2, 3)     #  1  2   3
    some_function(1)           #  1  42  'baz'
    some_function(1, bar=12)   #  1  42  12
    some_function()            #  TypeError: missing value for 'a'