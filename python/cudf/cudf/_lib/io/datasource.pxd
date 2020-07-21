# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.io.types cimport datasource

cdef class Datasource:
    cdef unique_ptr[datasource] c_datasource

    cdef inline datasource* get_datasource(self) except *:
        return <datasource *> self.c_datasource.get()
