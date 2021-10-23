

def storage(method):
    def wrapper(self, frequency: float):
        key = (frequency, method.__name__)
        if key not in self._storage:
            self._storage[key] = method(self, frequency)
        return self._storage[key]
    return wrapper
