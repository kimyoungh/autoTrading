"""
    상장 종목 종가 및 거래대금 시계열 가져와서 저장하는 스크립트

    @Date: 2020.02.28
    @Author: Younghyun Kim
"""

import datetime
import numpy as np
import pandas as pd

import cybos.cybosM as cm

class GetHistData:
    """
        상장 종목 데이터 가져오는 클래스
    """

    def __init__(self, load_codes=True):
        " initialization "

        self.today = datetime.datetime.now()
        self.prev = self.today - datetime.timedelta(days=1)

        self.today = int(self.today.strftime('%Y%m%d'))
        self.prev = int(self.prev.strftime('%Y%m%d'))

        if load_codes:
            self.codes = self.get_codes()
        else:
            self.codes = None

    def get_codes(self):
        " get codes "
        kospi = cm.get_stock_code_by_market('KOSPI')
        kosdaq = cm.get_stock_code_by_market('KOSDAQ')

        codes = kospi[kospi['Class'] == 'Stock'].index.values.tolist()
        codes += kosdaq[kosdaq['Class'] == 'Stock'].index.values.tolist()

        return codes

    def get_multiple_hist_data(self, codes, dtype='종가',
                               end_date=None, l=250, adj='1',
                               window=250):
        " 과거 데이터 Cybos Plus API를 통해 불러오는 함수 "
        data = cm.get_multiple_hist_data(codes, dtype=dtype,
                                         end_date=end_date, l=l+window, adj=adj)

        return data

if __name__ == "__main__":

    length = int(input("요청할 시계열 길이를 입력하시오: "))
    ghd = GetHistData(load_codes=True)
    price = ghd.get_multiple_hist_data(ghd.codes, dtype='종가',
                                       end_date=ghd.prev, l=length)
    turnover = ghd.get_multiple_hist_data(ghd.codes, dtype='거래대금',
                                          end_date=ghd.prev, l=length)
    mktcap = ghd.get_multiple_hist_data(ghd.codes, dtype='시가총액',
                                        end_date=ghd.prev, l=length)

    price.to_csv("./data/price_"+str(ghd.today)+".csv")
    turnover.to_csv("./data/turnover_"+str(ghd.today)+".csv")
    mktcap.to_csv("./data/mktcap_"+str(ghd.today)+".csv")
