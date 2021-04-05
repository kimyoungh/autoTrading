import sys
import datetime
import win32com.client as wc
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from datetime import datetime as dt


def get_code_by_name(name):
    """
        종목명을 입력 받아
        종목코드를 불러오는 함수
        Args:
        name: 종목 명
        Return:
        code: 종목코드
    """
    instCpStockCode = wc.Dispatch("CpUtil.CpStockCode")

    return instCpStockCode.NameToCode(name)

def get_all_stock_name():
    """
        등록된 전체 종목의 종목코드, 종목명 가져오기
        Return:
        stocks: dict / key: 종목코드, value: 종목명
    """
    cp_stock = wc.Dispatch("CpUtil.CpStockCode")
    stocks = {}

    i = 0
    bt = True
    while bt:
        try:
            code = cp_stock.GetData(0, i)
            name = cp_stock.GetData(1, i)
        except:
            bt = False
        finally:
            i += 1
            stocks[code] = name

    return stocks


def get_stock_code_by_market(market):
    """
        시장 별 종목 정보 가져오기
        Args:
            market:
                    1: KOSPI
                    2: KOSDAQ
                    3: KONEX
        Return:
            stocks: 종목 정보 DataFrame
    """
    if market == "KOSPI":
        m = 1
    elif market == "KOSDAQ":
        m = 2
    elif market == "KONEX":
        m = 3

    cp_code_mgr = wc.Dispatch("CpUtil.CpCodeMgr")

    code_list = cp_code_mgr.GetStockListByMarket(m)

    stocks = {}
    for _, code in enumerate(code_list):
        name = cp_code_mgr.CodeToName(code)
        secode = cp_code_mgr.GetStockSectionKind(code)
        if secode == 1:
            secode = "Stock"
        elif secode == 10:
            secode = "ETF"
        elif secode == 17:
            secode = "ETN"
        stocks[code] = [secode, name]

    stocks = pd.DataFrame(stocks, index=['Class', 'Name'])
    stocks = stocks.transpose()
    return stocks

def get_multiple_hist_data(codes, dtype='종가', end_date=None, c_type='D', l=250, adj='1'):
    """
        복수 종목 과거 종가 데이터 가져오는 함수
        Args:
            codes: 종목 코드 리스트
            dtype: 데이터 종류(기본값: 종가)
            end_date: 시계열 마지막 날짜
            c_type: 'D': Day, 'W': Week, 'M': Month, 'm': Minute, 'T': Tick
            l: 영업일 기준 시계열 길이(기본: 250일)
            adj: 수정주가 여부('1': 수정주가, '0': 무수정주가)
        Return:
            price: 복수 종목 주가 시계열 데이터프레임
    """
    data = {}
    length = []
    l_dict = {}
    dates = {}
    for i, code in enumerate(codes):
        p_data = get_hist_data(code, end_date=end_date, c_type=c_type, l=l, adj=adj)
        data[code] = p_data[dtype]
        dates[code] = p_data['날짜'].values
        length.append(data[code].shape[0])
        l_dict[i] = code

    length = np.array(length)
    l_max = np.argmax(length)
    max_code = l_dict[l_max]

    index_temp = dates[max_code]

    index = []
    for i, date in enumerate(index_temp):
        date_str = str(date)
        date_t = datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
        index.append(date_t)

    ptemp = pd.DataFrame(index=data[max_code].index)
    for code in data:
        ptemp[code] = data[code]

    price = pd.DataFrame(ptemp.values, columns=ptemp.columns, index=index)

    return price


def get_hist_data(code, start_date=None, end_date=None, c_type='D', l=None, adj='1'):
    """
        과거 데이터를 가져오는 함수
        code: 종목코드
        start_date, end_date: YYYYMMDD
        c_type: 'D': Day, 'W': Week, 'M': Month, 'm': Minute, 'T': Tick
        l: 시계열 길이
    """
    stock_chart = wc.Dispatch("CpSysDib.StockChart")

    # SetinputValue
    stock_chart.SetInputValue(0, code)
    if l is not None:
        stock_chart.SetInputValue(1, ord('2'))
        stock_chart.SetInputValue(4, l)
    else:
        stock_chart.SetInputValue(1, ord('1'))
        stock_chart.SetInputValue(2, start_date)
        stock_chart.SetInputValue(3, end_date)
    stock_chart.SetInputValue(5, (0, 1, 2, 3, 4, 5, 8, 9, 13, 19))
    stock_chart.SetInputValue(6, ord(c_type))
    stock_chart.SetInputValue(9, ord(adj))

    stock_chart.BlockRequest()

    # GetHeaderValue
    numData = stock_chart.GetHeaderValue(3)
    numField = stock_chart.GetHeaderValue(1)

    data = []
    # GetDataValue
    for i in range(numData):
        temp = []
        for j in range(numField):
            temp.append(stock_chart.GetDataValue(j, i))
        data.append(temp)
    
    data = pd.DataFrame(data, columns=['날짜', '시간', '시가', '고가',
                                       '저가', '종가', '거래량',
                                       '거래대금',
                                       '시가총액', '수정주가비율'])
    data = data.sort_values(by=['날짜', '시간'])

    return data

def get_bid_ask(code):
    bid_ask = wc.Dispatch("Dscbo1.StockJpBid2")

    bid_ask.SetInputValue(0, code)
    bid = {}
    ask = {}

    bid_ask.BlockRequest()

    for i in range(10):
        temp = {}
        temp['매수호가'] = bid_ask.GetDataValue(1, i)
        temp['매수잔량'] = bid_ask.GetDataValue(3, i)
        bid['bid_'+str(i)] = temp

        temp = {}
        temp['매도호가'] = bid_ask.GetDataValue(0, i)
        temp['매도잔량'] =  bid_ask.GetDataValue(2, i)
        ask['ask_'+str(i)] = temp

    return bid, ask


def cur_price(code):
    cur = wc.Dispatch("Dscbo1.StockMst")

    cur.SetInputValue(0, code)

    cur.BlockRequest()
    price = cur.GetHeaderValue(11)

    return price

def priceData(code, l=250):
    """
        과거 가격 데이터 가져오는 메소드
        code: 주식 코드(ex. A005930)
        l: 시계열 길이(영업일 기준, 기본값: 250일)

        결과: 날짜별로 시초가, 최고가, 최저가, 종가, 거래량 시계열 담은
        데이터프레임
    """

    obj = wc.Dispatch("CpUtil.CpCybos")
    bConnect = obj.IsConnect
    assert(bConnect == 1)

    objs = wc.Dispatch("DsCbo1.StockWeek")
    objs.SetInputValue(0, code)  # 종목코드 입력

    _objload(objs)

    c = objs.GetHeaderValue(1)  # 데이터 개수
    k = l // c
    m = l % c

    date = []
    open = []
    high = []
    low = []
    close = []
    vol = []
    tover = []

    if k > 0:
        for j in range(c):
            d = objs.GetDataValue(0, j)
            try:
                d = str(d)
                d = dt(int(d[:4]), int(d[4:6]), int(d[6:])).date()
            except:
                d = np.nan
            date.append(d)
            open.append(objs.GetDataValue(1, j))
            high.append(objs.GetDataValue(2, j))
            low.append(objs.GetDataValue(3, j))
            close.append(objs.GetDataValue(4, j))
            vol.append(objs.GetDataValue(6, j))
            tover.append(objs.GetDataValue(20, j))

        for i in range(k-1):
            _objload(objs)

            for j in np.arange(c):
                d = objs.GetDataValue(0, j)
                try:
                    d = str(d)
                    d = dt(int(d[:4]), int(d[4:6]), int(d[6:])).date()
                except:
                    d = np.nan
                date.append(d)
                open.append(objs.GetDataValue(1, j))
                high.append(objs.GetDataValue(2, j))
                low.append(objs.GetDataValue(3, j))
                close.append(objs.GetDataValue(4, j))
                vol.append(objs.GetDataValue(6, j))
                tover.append(objs.GetDataValue(20, j))
    if k > 0:
        _objload(objs)

    for i in range(m):
        d = objs.GetDataValue(0, i)
        try:
            d = str(d)
            d = dt(int(d[:4]), int(d[4:6]), int(d[6:])).date()
        except:
            d = np.nan
        date.append(d)
        open.append(objs.GetDataValue(1, i))
        high.append(objs.GetDataValue(2, i))
        low.append(objs.GetDataValue(3, i))
        close.append(objs.GetDataValue(4, i))
        vol.append(objs.GetDataValue(6, i))
        tover.append(objs.GetDataValue(20, i))

    assert(len(date) == len(open) == len(high))
    assert(len(high) == len(low) == len(close) == len(vol))
    assert(len(vol) == len(tover))

    price = df(index=date)
    price['Open'] = open
    price['High'] = high
    price['Low'] = low
    price['Close'] = close
    price['Vol'] = vol
    price['Turnover'] = tover

    index = price.index
    index = index.sort_values()
    price = price.loc[index]

    return price


def _objload(obj):
    """
        과거 연속 데이터 불러올때 사용하는 내부 함수
    """
    obj.BlockRequest()

    # 통신 결과 확인
    rqStatus = obj.GetDibStatus()
    rqRet = obj.GetDibMsg1()
    assert(rqStatus == 0)


def multiPriceData(codes, l=250):
    """
        복수 종목 시계열 데이터 가져오는 함수
        codes: 종목코드 모음(ex. [A005930, A004930])
        l: 시계열길이(영업일 기준, 기본값: 250일)
        return: 시초가, 최고가, 최저가, 종가, 거래량, 거래대금
    """
    data = {}
    ldata = []

    for code in codes:
        temp = priceData(code, l=l)
        ldata.append(len(temp))
        data[code] = temp

    ldata = np.array(ldata)
    lmax = np.argmax(ldata)
    st = list(data.keys())[lmax]
    cols = data[codes[0]].columns

    date = data[st].index
    open = df(index=date)
    high = df(index=date)
    low = df(index=date)
    close = df(index=date)
    vol = df(index=date)
    tover = df(index=date)

    for code in codes:
        open[code] = data[code][cols[0]]
        high[code] = data[code][cols[1]]
        low[code] = data[code][cols[2]]
        close[code] = data[code][cols[3]]
        vol[code] = data[code][cols[4]]
        tover[code] = data[code][cols[5]]

    return open, high, low, close, vol, tover


if __name__ == "__main__":
    # 삼성전자, 카카오, KT&G, 휠라코리아
    scodes = ['A005930', 'A035720', 'A033780', 'A081660']
    oprice, hprice, lprice, close, vol, tover = multiPriceData(scodes, l=750)
