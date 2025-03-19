import sys
import time
import datetime as dt
from unittest import signals
from tqdm import tqdm
import matplotlib as mpl
import pandas as pd
import numpy as np
import copyreg, types, multiprocessing as mp


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def get_previous_dates(close_prices):
    previous_day_index = close_prices.index.searchsorted(close_prices.index - pd.Timedelta(days=1))
    is_valid = previous_day_index > 0
    return pd.Series(
        close_prices.index[previous_day_index[is_valid] - 1],
        index=close_prices.index[-is_valid.sum():]
    )


def getDailyVol(close, span=100):
    """
    Calculate daily volatility using an Exponentially Weighted Moving Standard Deviation.

    CHECK FOR DULPICATES!!!!!!!!!!

    Parameters:
    close (pd.Series): Time series of prices with a DatetimeIndex.
    span (int): The span of the EWM (exponentially weighted mean).

    Returns:
    pd.Series: Daily volatility values.
    """

    # Ensure the input series is sorted by time
    close = close.sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("`close` must have a DatetimeIndex.")

    # Calculate time delta (1-day difference)
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]  # Filter valid indices

    # Handle potential mismatches in indices
    try:
        df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
        # Calculate returns over the 1-day window
        daily_returns = close.loc[df0.index] / close.loc[df0.values].values - 1
    except KeyError as e:
        raise KeyError("Mismatch in indices during return calculation.") from e

    # Apply EWM for standard deviation (volatility)
    daily_vol = daily_returns.ewm(span=span).std()
    return daily_vol


# def get_daily_volatility(close_prices, span=100):
#     sorted_close = close_prices.sort_index()
#     previous_dates = get_previous_dates(sorted_close)
#     daily_returns = sorted_close.loc[previous_dates.index] / sorted_close.loc[previous_dates.values].values - 1
#     volatility = daily_returns.ewm(span=span).std()
#     return volatility

def getVb(data, events):
    t1 = data.index.searchsorted(events + pd.Timedelta(days=1))
    t1 = t1[t1 < data.shape[0]]
    t1 = pd.Series(data.index[t1], index=events[:t1.shape[0]])
    return t1


def getTEvents(gRaw, h):  ###cusum filter
    tEvents, sPos, sNeg = [], 0, 0
    # h=h*gRaw.mean()
    diff = gRaw.pct_change()
    # diff=gRaw
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def applyPtSlOnT1(close, events, ptSl, molecule):
    # print(molecule)
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        # print(">0)")
        # print("pstl[0] is:",ptSl[0])
        pt = ptSl[0] * events_['trgt']
        # print("pt is:",pt)
    else:
        pt = pd.Series(index=events.index)  # NaNs
        # print(pt)
    if ptSl[1] > 0:
        # print("<1)")
        # print("ptsl[1] is:",ptSl[1])
        sl = -ptSl[1] * events_['trgt']
        # print("pt is:",pt)
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.reindex(tEvents, method='bfill')  # added line
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close,
                      events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None: events = events.drop('side', axis=1)
    return events


def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    —events.index is event's starttime
    —events[’t1’] is event's endtime
    —events[’trgt’] is event's target
    —events[’side’] (optional) implies the algo's position side
    Case 1: (’side’ not in events): bin in (-1,1) <—label by price action
    Case 2: (’side’ in events): bin in (0,1) <—label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3: break
        print('dropped label', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events


def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are the heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    '''
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func
    Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
    '''

    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):  # xrange replace by range in python3
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
        # print("job ",i," of ",len(parts)," done")
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    if isinstance(out[0], pd.DataFrame) or isinstance(out[0], pd.Series):
        df0 = pd.concat(out)
    return df0.sort_index()


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    print("Running processJobs_")
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def movingAverageCrossover(prices, short_window=20, long_window=50):
    """
    ChatGPT generated code:
    Calculate moving average crossover signals.

    Parameters:
    - prices (pd.Series): Series of price data.
    - short_window (int): Window size for the short-term moving average.
    - long_window (int): Window size for the long-term moving average.

    Returns:
    - pd.DataFrame: DataFrame with short-term and long-term moving averages and crossover signals.
    """
    short_ma = prices.rolling(window=short_window, min_periods=1).mean()
    long_ma = prices.rolling(window=long_window, min_periods=1).mean()

    signals = pd.DataFrame(index=prices.index)
    signals['short_ma'] = short_ma
    signals['long_ma'] = long_ma
    signals['signal'] = 0
    signals.loc[short_ma > long_ma, 'signal'] = 1
    signals.loc[short_ma <= long_ma, 'signal'] = -1

    return signals
    # return pd.DatetimeIndex(signals.index)


def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    print("Running processJobs")
    # jobs must contain a ’func’ callback, for expandCall
    if task is None: task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        # print(i)
        reportProgress(i, len(jobs), time0, task)
    pool.close();
    pool.join()  # this is needed to prevent memory leaks
    return out


def expandCall(kargs):
    # Expand the arguments of a callback function, kargs[’func’]
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def getRolledSeries(series):
    # series = pd.read_hdf(pathIn, key='bars/ES_10k')
    # series['Time'] = pd.to_datetime(series['Time'], format='%Y%m%d%H%M%S%f')
    series = series.set_index('Time')
    gaps = rollGaps(series)
    for fld in ['Close', 'Open', 'High', 'Low']: series[fld] -= gaps
    return series


def rollGaps(series, dictio={'Instrument': 'Symbol', 'Open': 'Open', \
                             'Close': 'Close'}, matchEnd=True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates]  # index of days prior to roll
    gaps.loc[rollDates[1:]] = series[dictio['Open']].loc[rollDates[1:]] - \
                              series[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd: gaps -= gaps.iloc[-1]  # roll backward
    return gaps


def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.

    My notes:
    closeIdx gives us the bars that we use to compute the number of events for, so it is the prices.

    '''
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items(): count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    # SNIPPET 4.2 ESTIMATING THE AVERAGE UNIQUENESS OF A LABEL
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getIndMatrix(barIx, t1):
    # SNIPPET 4.3 BUILD AN INDICATOR MATRIX
    # Get indicator matrix
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.items()): indM.loc[t0:t1, i] = 1.
    return indM


def getAvgUniqueness(indM):
    # SNIPPET 4.4 COMPUTE AVERAGE UNIQUENESS
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU


def seqBootstrap(indM, sLength=None):
    # SNIPPET 4.5 RETURN SAMPLE FROM SEQUENTIAL BOOTSTRAP
    # Generate a sample via sequential bootstrap
    if sLength is None: sLength = indM.shape[1]
    phi = []
    with tqdm(total=sLength, desc="Sequential Bootstrap") as pbar:
        while len(phi) < sLength:
            avgU = pd.Series()
            for i in indM:
                indM_ = indM[phi + [i]]  # reduce indM
                avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
            prob = avgU / avgU.sum()  # draw prob
            phi += [np.random.choice(indM.columns, p=prob)]
            pbar.update(1)
    return phi


def mpSampleW(t1, numCoEvents, close, molecule):
    # SNIPPET 4.10 DETERMINATION OF SAMPLE WEIGHT BY ABSOLUTE RETURN ATTRIBUTION
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW, clfLastW=1.):
    # SNIPPET 4.11 IMPLEMENTATION OF TIME-DECAY FACTORS
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print
    const, slope
    return clfW


def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
    w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
    w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
    w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left');
    mpl.show()
    return


def fracDiff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]): continue  # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

