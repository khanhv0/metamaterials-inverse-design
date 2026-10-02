from PySide6 import QtCore
from pandas import DataFrame

class ObservableValue(QtCore.QObject):
    hasChangedSignal = QtCore.Signal(object)

    def __init__(self, val = None):
        super().__init__()
        self._value = val

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_val):
        is_df = isinstance(self._value, DataFrame)
        if self._value is None or (is_df and not self._value.equals(new_val) 
                            or not is_df and self._value != new_val) :

            self._value = new_val
            self.hasChangedSignal.emit(new_val)