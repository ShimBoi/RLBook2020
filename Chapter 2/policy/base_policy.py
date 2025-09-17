from abc import ABC, abstractmethod

class BasePolicy:

    @abstractmethod
    def action(self):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, action, reward):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError