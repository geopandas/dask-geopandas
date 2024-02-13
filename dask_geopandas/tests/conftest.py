import dask


# TODO Disable usage of pyarrow strings until the expected results in the tests
# are updated to use those as well
dask.config.set({"dataframe.convert-string": False})
