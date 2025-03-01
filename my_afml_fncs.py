import sys
import time
import datetime as dt
from unittest import signals

import pandas as pd
import numpy as np
import copyreg, types, multiprocessing as mp

def _pickle_method(method):
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method,(func_name,obj,cls)

def _unpickle_method(func_name,obj,cls):
    for cls in cls.mro():
        try:
            func=cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj,cls)

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def get_previous_dates(close_prices):
    previous_day_index = close_prices.index.searchsorted(close_prices.index - pd.Timedelta(days=1))
    is_valid = previous_day_index > 0
    return pd.Series(
        close_prices.index[previous_day_index[is_valid] - 1],
        index=close_prices.index[-is_valid.sum():]
    )

def getDailyVol(close, span=100):
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1],index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    df0=df0.ewm(span=span).std()
    return df0

# def get_daily_volatility(close_prices, span=100):
#     sorted_close = close_prices.sort_index()
#     previous_dates = get_previous_dates(sorted_close)
#     daily_returns = sorted_close.loc[previous_dates.index] / sorted_close.loc[previous_dates.values].values - 1
#     volatility = daily_returns.ewm(span=span).std()
#     return volatility

def getVb(data,events):
    t1 = data.index.searchsorted(events + pd.Timedelta(days=1))
    t1 = t1[t1 < data.shape[0]]
    t1 = pd.Series(data.index[t1], index=events[:t1.shape[0]])
    return t1

def getTEvents(gRaw,h):          ###cusum filter
    tEvents,sPos,sNeg=[],0,0
    # h=h*gRaw.mean()
    diff=gRaw.pct_change()
    # diff=gRaw
    for i in diff.index[1:]:
        sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0
            tEvents.append(i)
        elif sPos>h:
            sPos=0
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

def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False,side=None):
    #1) get target
    trgt = trgt.reindex(tEvents, method='bfill') #added line
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index),[ptSl[0],ptSl[0]]
    else:side_,ptSl_=side.loc[trgt.index],ptSl[:2]
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt'])
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index), numThreads=numThreads,close=close,events=events,ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None: events = events.drop('side', axis=1)
    return events

def getBins(events,close):
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
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

def dropLabels(events,minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:break
        print('dropped label',df0.argmin(),df0.min())
        events=events[events['bin']!=df0.idxmin()]
    return events

def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are the heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func
    Example: df1=mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
    '''

    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs=[]
    for i in range(1,len(parts)): #xrange replace by range in python3
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
        # print("job ",i," of ",len(parts)," done")
    if numThreads==1:out=processJobs_(jobs)
    else:out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    if isinstance(out[0], pd.DataFrame) or isinstance(out[0], pd.Series):
        df0 = pd.concat(out)
    return df0.sort_index()

def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    print("Running processJobs_")
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum,numJobs,time0,task):

    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
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

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    print("Running processJobs")
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        # print(i)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs[’func’]
    func=kargs['func']
    del kargs['func']
    out=func(** kargs)
    return out

def getRolledSeries(series):
    # series = pd.read_hdf(pathIn, key='bars/ES_10k')
    # series['Time'] = pd.to_datetime(series['Time'], format='%Y%m%d%H%M%S%f')
    series = series.set_index('Time')
    gaps = rollGaps(series)
    for fld in ['Close','Open','High','Low']: series[fld] -= gaps
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
