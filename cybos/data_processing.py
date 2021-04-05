from get_hist_data import GetHistData
from calculate_multifactors import CalculateMultifactors

length = int(input("요청할 시계열 길이를 입력하시오: "))
ghd = GetHistData(load_codes=True)

price = ghd.get_multiple_hist_data(ghd.codes, dtype='종가',
                                   end_date=ghd.prev, l=length)
turnover = ghd.get_multiple_hist_data(ghd.codes, dtype='거래대금',
                                      end_date=ghd.prev, l=length)
mktcap = ghd.get_multiple_hist_data(ghd.codes, dtype='시가총액',
                                    end_date=ghd.prev, l=length)

price_name = "./data/price_"+str(ghd.today)+".csv"
turnover_name = "./data/turnover_"+str(ghd.today)+".csv"
mktcap_name = "./data/mktcap_"+str(ghd.today)+".csv"

price.to_csv(price_name)
turnover.to_csv(turnover_name)
mktcap.to_csv(mktcap_name)

cmf = CalculateMultifactors(price_name, turnover_name, mktcap_name)

cmf.calculate_dataset()
