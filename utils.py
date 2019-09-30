def yield_batches(iterable, batch_size):
    """ Splits the data into batches using yield
    
    Arguments:
        iterable {function} -- range(start,end)
        batch_size {int} -- The size of each batch

        Example: for x in batches(range(0, 10), 100):
    """
    endpoint = len(iterable)
    for ndx in range(0, endpoint, batch_size):
        yield iterable[ndx:min(ndx + batch_size, endpoint)]


