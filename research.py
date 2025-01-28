import numpy as np
import pandas as pd
import datetime as dt
import warnings
from sklearn.datasets import make_classification
from Util.multiprocess import mp_pandas_obj

def create_price_data(start_price: float = 1000.00, mu: float = .0, var: float = 1.0, n_samples: int = 1000000):
    i = np.random.normal(mu, var, n_samples)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.Minute(), end=dt.datetime.today())
    X = pd.Series(i, index=df0, name="close").to_frame()
    X.at[X.index[0], 'close'] = start_price
    X.cumsum().plot.line()
    return X.cumsum()


def make_randomt1_data(n_samples: int = 10000, max_days: float = 5., Bdate: bool = True):
    # generate a random dataset for a classification problem
    if Bdate:
        _freq = pd.tseries.offsets.BDay()
    else:
        _freq = 'D'
    _today = dt.datetime.today()
    df0 = pd.date_range(periods=n_samples, freq=_freq, end=_today)
    rand_days = np.random.uniform(1, max_days, n_samples)
    rand_days = pd.Series([dt.timedelta(days=d) for d in rand_days], index=df0)
    df1 = df0 + pd.to_timedelta(rand_days, unit='d')
    df1.sort_values(inplace=True)
    X = pd.Series(df1, index=df0, name='t1').to_frame()
    return X


def make_classification_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, days: int = 1):
    # generate a random dataset for a classification problem
    _today = dt.datetime.today()
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.BDay(), end=_today)
    X = pd.DataFrame(X, index=df0)
    y = pd.Series(y, index=df0).to_frame('bin')
    df0 = ['I_%s' % i for i in range(n_informative)] + ['R_%s' % i for i in range(n_redundant)]
    df0 += ['N_%s' % i for i in range(n_features - len(df0))]
    X.columns = df0
    y['w'] = 1.0 / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index - dt.timedelta(days=days))
    y.at[-1:, 't1'] = _today
    y.t1.fillna(method='bfill', inplace=True)
    return X, y


def create_portfolio(price: list = [95, 1000], position: list = [1000, 10000], n_sample: int = 10000,
                     imbalance: bool = True):
    df = make_randomt1_data(n_samples=n_sample, max_days=10., Bdate=True)
    df['open'] = np.random.uniform(price[0], price[1], n_sample)
    df['close'] = df['open'].apply(lambda x: x * np.random.uniform(0.8, 1.2))
    df['open_pos'] = np.random.uniform(position[0], position[1], n_sample).round()
    df['close_pos'] = np.random.uniform(position[0], position[1], n_sample).round()

    df['yield'] = np.random.uniform(0.012, 0.12, n_sample)
    df['expense'] = df['open'].apply(lambda x: x * np.random.uniform(0.005, 0.05))
    df['label'] = np.nan
    if imbalance:
        for idx in df[df['open'] < 200].index: df.loc[idx, 'label'] = 1
    else:
        for idx in df[df['open'] < df['open'].quantile(0.4)].index: df.loc[idx, 'label'] = 1
    df['label'].fillna(0, inplace=True)

    df['clean_open'] = df['open'].apply(lambda x: x * np.random.uniform(0.92, 0.95))
    df['clean_close'] = df['yield']
    df['clean_close'] = df['clean_close'].apply(lambda x: 1 - x).mul(df['close'])
    df['yield'] = df['yield'].mul(df['close'])

    p_str = "Sample Portfolio Construct:\n{0}\nEquity label: {1}\nBond label: {2}\nEquity to Debt Ratio: {3:.4f}"
    p(p_str.format("=" * 55,
                   df['label'].value_counts()[0],
                   df['label'].value_counts()[1],
                   df['label'].value_counts()[0] / df['label'].value_counts()[1]))

    junk_bond = df[(df['yield'] >= 0.1) & (df['label'] == 1)].count()[0]
    div_equity = df[(df['yield'] >= 0.1) & (df['label'] != 1)].count()[0]
    p("\nJunk bond (Below BBB-grade): {0} %\nDividend equity: {1} %".format(100 * junk_bond / n_sample,
                                                                            100 * div_equity / n_sample))
    return df


def create_rtndf(n_samples: int = 1000, max_days: int = 5, rtn: list = [-.5, .5], Bdate: bool = True):
    if rtn[0] >= rtn[1] or rtn[1] < 0:
        rtn = [-.5, .5]
    df = make_randomt1_data(n_samples=n_samples // 2, max_days=max_days, Bdate=Bdate)
    idx = pd.to_datetime(df.index.union(df.t1).sort_values())
    df = pd.DataFrame(1., index=idx, columns=np.arange(n_samples))
    df = df.applymap(lambda x: np.random.uniform(rtn[0], rtn[1]))
    return df


def vert_barrier(data: pd.Series, events: pd.DatetimeIndex, period: str = 'days', freq: int = 1):
    '''
    AFML pg 49 modified snippet 3.4
    This is not the original snippet, there is some slight change to period

    params: data => close price
    params: period => weeks, days, hours, mins
    params: freq => frequency i.e. 1

    Vertical barrier will be events_'t1', which will be part of the final func
    where based on filtered criteria (i.e. sys_csf) will generate a period using vert_bar based on freq and period input
    the column is t1 is to maintain consistance so that we can continue to use in conjunction with other modules i.e. co_events

    This func does not include holiday and weekend exclusion.
    Strongly encourage you to ensure your series does not include non-trading days.
    As well as after trading hours, to prevent OMO.

    Otherwise you may end up having non-trading days as an as exit event where it could exceed initial parameter.
    i.e. trigger event on Fri and exit on Monday when vertical barrier only 1 day.

    Another note:
    If you are using information driven bars or any series that are not in chronological sequence.
    This func will automatically choose the nearest date time index which may result in more than intended period.
    '''
    if isinstance(data, (int, float, str, list, dict, tuple)):
        raise ValueError('data must be pandas series with DatetimeIndex i.e. close price series')
    elif isinstance(data.squeeze().dtype, (str, list, dict, tuple)):
        raise ValueError('data dtype must be integer or float value i.e. 1.0, 2')
    elif data.index.dtype != 'datetime64[ns]':
        raise ValueError('data index does not contain datetime')

    if isinstance(events, (int, float, str, list, dict, tuple)):
        raise ValueError('events must be pandas DatetimeIndex')
    elif events.dtype != 'datetime64[ns]':
        raise ValueError('events must be pandas DatetimeIndex')

    if isinstance(period, (pd.Series, np.ndarray, list, int, float)):
        raise ValueError('Period should be string i.e. days')
    elif period != 'days':
        warnings.warn('Recommend using days for simplicity')

    if isinstance(freq, (pd.Series, np.ndarray, list, str, float)):
        raise ValueError('Frequency must be in integer, other dtypes not accepted i.e. float, np.ndarray')
    elif freq <= 0:
        raise ValueError('Frequency must be in positive integer')

    _period = str(freq) + period
    t1 = data.index.searchsorted(events + pd.Timedelta(_period))
    t1 = t1[t1 < data.shape[0]]
    t1 = pd.Series(data.index[t1], index=events[:t1.shape[0]])
    return t1


def vol(data: pd.Series, span0: int = 100, period: str = 'days', num_period: int = 1):
    '''
    AFML page 44. 3.1 snippet
    Modify from the original daily volatility

    This will retrieve an estimate of volatility based on initial params
    The modification is to track stablility count etc and for other research

    param: pd.Series => data use close price
    param: int => num of samples for ewm std
    param: datetime/ string => specify Day, Hour, Minute, Second, Milli, Micro, Nano
    param: int => frequency, integer only
    '''
    if isinstance(data, (str, int, float)):
        raise ValueError("data must be pandas series, 1d array with datetimeIndex i.e close price series")

    if isinstance(span0, (str, list, pd.Series, np.ndarray) or span0 <= 0):
        raise ValueError("span0 must be non-zero positive integer or float i.e 21 or 7.0")

    if isinstance(period, (float, int, list, pd.Series, np.ndarray)):
        raise ValueError("period must string i.e 'days', 'mins'")

    if isinstance(num_period, (str, list, pd.Series, np.ndarray)) or num_period <= 0:
        raise ValueError("num_period must non-zero positive integer i.e 100, 50")
    else:
        num_period = int(num_period)

    freq = str(num_period) + period
    df0 = data.index.searchsorted(data.index - pd.Timedelta(freq))
    df0 = df0[df0 > 0]
    df0 = pd.Series(data.index[df0 - 1], index=data.index[data.shape[0] - df0.shape[0]:])
    df0 = data.loc[df0.index] / data.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


def tri_barrier(data: pd.Series, events: pd.DatetimeIndex, trgt: pd.Series, min_req: float,
                num_threads: int = 3, ptSl: list = [1, 1], t1: pd.Series = False, side: pd.Series = None):
    '''
    AFML pg 50 snippet 3.6
    This function will return triple barrier data based on params.

    There is some amendment to the original snippet provided, to prevent error.
    Added side_ variable which act as boolean counter: 0 = No side prediction provided.(No primary model)
                                                       1 = Side prediction provided. (Primary model)

    Some amendment to Pandas multiprocessor object, due to the my machine limitation.
    Default thread is 1 for all func that requires multiprocessing.
    Pandas multiprocessor Pool contains a min 1 threads, so default is 2 threads.

    ptSl is upper and lower limit of spot price.
    Hence to apply additional multipler effect provide a positive value more than 1

    params: data => close price series
    params: events => Datetime series sys_cusum filter always
    params: trgt => target from sample pct change
    params: min_req => minimal requirement (Recommend: transaction cost as a percentage)
    last 4 params: contains default values
                    optional params: num_threads => integer this is for multiprocessing per core
                    optional params: ptSl => list(), [] width of profit taking and stop loss
                    optional params: t1 => pd.DataFrame for vertical_bar()
                    optional params: side => pd.Series() side column must be setup based on primary model

    '''
    if isinstance(data, (str, float, int, dict, tuple)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. close price series')

    if isinstance(events, (str, float, int, dict, tuple)):
        raise ValueError('Data must be pandas DatetimeIndex i.e. pd.DatetimeIndex series')
    elif events.shape[0] != data.index.shape[0]:
        warnings.warn('Data and events index shape must be same, reindex data to fit events')
    else:
        isinstance(events, datetime.datetime)

    if isinstance(trgt, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. sample data percentage change series')

    # Optional params test
    if isinstance(num_threads, (str, float, list, dict, tuple, np.ndarray, pd.Series)) or (num_threads < 0):
        raise ValueError('num_threads must be non-zero postive integer i.e. 2')

    if isinstance(ptSl, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or list i.e. [1,1]')
    elif ptSl[0] == np.nan or ptSl[1] == np.nan:
        raise ValueError('Data must be numpy 1darray shape(1,2) i.e. [1,1]')
    elif ptSl[0] < 0 or ptSl[1] < 0:
        # test case for irrational users
        raise ValueError('Data must be numpy 1darray shape(1,2) with values more than 0 i.e. [1,1]')

    if isinstance(t1, (pd.Series)):
        if t1.isnull().values.any():
            raise ValueError('t1 cannot have NaTs, pls use vertical_bar func provided.')
        elif isinstance(t1.dtype, (str, float, int, list, dict, tuple)):
            raise ValueError('t1 must be pd.Series with datetime values, pls use vertical_bar func provided.')
        elif t1.dtype != 'datetime64[ns]':
            raise ValueError('t1 must be pd.Series with datetime, pls use vertical_bar func provided.')
    elif t1 == False:
        warnings.warn('\nNot Recommended: No vertical barrier provided')

    if isinstance(side, (pd.Series)):
        if side.isnull().values.any():
            raise ValueError('side must be pd.Series based on primary model prediction w/o NaNs.')
        elif isinstance(side.dtype, (str, list, dict, tuple)):
            raise ValueError(
                'side must be pd.Series based on primary model prediction with integer or float values i.e. (-1,1), (0,1), (-1,0)')
        elif side.max() > 1 or side.min() < -1:
            raise ValueError('side must be pd.Series based on primary model prediction with values range(-1, 1).')
    elif side == None:
        warnings.warn('Not Recommended: No side prediction provided')

    data = pd.DataFrame(index=events).assign(data=data).squeeze()  # recreate data based on index
    trgt = trgt.reindex(events)
    trgt = trgt[trgt > min_req]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=events)
    if side is None:
        side_, side, ptSl_ = 0, pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]  # create side_ as counter
    else:
        side_, side, ptSl_ = 1, side.reindex(trgt.index), ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side}, axis=1).dropna(subset=['trgt'])
    df0 = mp_pandas_obj(func=_pt_sl_t1,
                        pd_obj=('molecule', events.index),
                        num_threads=num_threads,
                        data=data,
                        events=events,
                        ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignore NaNs
    if side_ == 0:
        events = events.drop('side', axis=1)  # if side_ counter is set to 0
    return events.dropna()


def cs_filter(data: pd.Series, limit: float):
    '''
    params: pd.Series => time series input price only accepts ndarray, list, pd.series
    params: pd.Series => threshold before logging datetime index

    AFML pg 39 snippet 2.4
    This func will give absolute return based on input price.
    As for limit, use original price series to derive standard deviation as an estimate.
    This is to ensure stationary series is homoscedastic.

    Logic is the same, but kindly go through the code before using it.

    WARNING!!: DO NOT EVER CHANGE data.diff() into data.apply(np.log).diff(), use only absolute price.

    np.log(data) does allow memory preservation but! that is not what we are looking for.

    In addition, np.log(data) contains additive properties. This will distort your data structure.

    The main use use for this func is to ensure your data structure is homoscedastic as much as possible, which is pivotal for mean-reversion strategy.
    IF you change this line, 99.99% you will only get a Heteroscedastic data structure, no matter what you do at a later stage.

    This will haunt you at the later stage as you develop your mean-reversion strategy.

    The above claim is tested and proven (sort of..)

    This filter will ensure your data structure maintain a good data structural shape ,
    which is key to mean-reversion strategy and to ensure your data structure is NOT too random.

    If you are not sure  what I mean pls go read up on "time-series stationarity" and run a white test using both log price and abs price.

    REPEAT 10 times: "you will not change this func"

    This filter return datatimeindex only.
    '''
    if isinstance(data, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')

    if isinstance(limit, (list, np.ndarray, pd.Series)):
        limit = float(limit.mean())
        UserWarning('Func does not accept numpy array, convert limit to mean value as estimate')
    elif isinstance(limit, (int, float)):
        limit = float(limit)
        UserWarning('Convert limit to float value as estimate')
    else:
        raise ValueError('Limit is neither numpy ndarray, pandas series nor float!')

    idx, _up, _dn = [], 0, 0
    diff = data.diff()
    for i in diff.index[1:]:
        _up, _dn = max(0, float(_up + diff.loc[i])), min(0, float(_dn + diff.loc[i]))
        if _up >= limit:
            _up = 0;
            idx.append(i)
        elif _dn <= - limit:
            _dn = 0;
            idx.append(i)

    return pd.DatetimeIndex(idx)


def _pt_sl_t1(data: pd.Series, events: pd.Series, ptSl: list, molecule):
    '''
    AFML pg 45 snippet 3.2

    Code snippet which tells you the logic of how a triple barrier will be formed.
    In the event if price target is touched first before vertical barrier.
    The func is created to be run with python multiprocesses module

    params: data => closing price
    params: events => new dataframe with timestamp index, target, metalabel
    params: ptSl => an array [1,1] which will determine width of horizontal barriers
    params: data => molecule which is part of multiprocess module, to break down jobs and allow parallelisation

    '''
    events_ = events.reindex(molecule)
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # Series with index but no value NaNs
    if ptSl[1] > 0:
        sl = - ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # Series with index but no value NaNs
    for loc, t1 in events_['t1'].fillna(data.index[-1]).items():
        df0 = data[loc:t1]  # if tri_bar does not create new assign dataframe, data will go haywire when events
        df0 = (df0 / data[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out


def meta_label(data: pd.Series, events: pd.DataFrame, drop=False):
    '''
    AFML page 51 snippet 3.7
    Basically this func will return meta-labels or price-labels, which is dependent on 'side' if exist.

    If meta-label used it return boolean values (0,1) where 0 means not profitable or vertical barrier was hit first
    and 1 would mean profitable.

    This method will complement a primary model with trading rules, and act as a secondary model.

    Note:
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling/ recommended)

    params: data => close price which will be used to compare if targets were profitable
    params: events = > DataFrame from triple barrier func
                        -events.index is event's starttime
                        -events['t1'] is event's endtime
                        -events['trgt'] is event's target
                        -events['side'] (optional) implies the algo's position side

    '''
    if isinstance(data, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. close price series')
    if isinstance(events, (int, float, str, list, dict, tuple)):
        raise ValueError(
            'Data must be pd.DataFrame, this function is used after triple barrier function i.e. close price series')

    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = data.reindex(px, method='bfill')

    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        out['side'] = events_['side']
    if drop is not False:
        out = drop_label(events=out, min_pct=drop)
    return out


# =======================================================
# Labeling for side and size [3.5, 3.8]

def drop_label(events: pd.Series, min_pct: float = .05):
    # apply weights, drop labels with insufficient examples
    '''
    Drop labels with insufficient example
    During training stage only labels deem somewhat reliable will be used

    Normalized outcome as a measure, to maintain shape of data sample.
    Default critical size is 0.05, basically if a label is consider rare it will be dropped.
    Increasing critical size will impact the number of data sample available, but also if too many rare sample is fed to ML.
    It will reduce accuracy.

    If the based on value count of 'bin', lowest frequency is 0. Then 0 will be dropped entirely.
    The criteria for this infinite loop to break would be to drop one of the labels -1,0,1
    Or minimal percentage is less than bin value count this is to prevent imbalance within data sample.


    params: events => DataFrame from labels func
    params: min_pct => float value as a measure based on normalization.
                       This is not return measure but rather what is the lowest percentage to accpet as non-rare labels.

    '''
    if isinstance(min_pct, (list, np.ndarray, pd.Series)) or min_pct <= 0:
        raise ValueError('min_pct must be positive float i.e. 0.05')
    elif min_pct > 1:
        raise ValueError('min_pct must be within range(0,1) i.e. 0.05')

    if isinstance(events, (int, float, str, list, dict, tuple)):
        raise ValueError('events must be pd.DataFrame, kindly use label func provided')

    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events