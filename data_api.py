import yfinance as yf


def download_ticker_data(tickers, start_date, end_date, interval='1mo'):
    """
    Use this function to download ticker data and change the source of the data if necessary
    Currently using yfinance
    :param tickers: string or list of strings representing ticker symbols of securities traded on public exchanges.
    :param start_date: string representing the start date of the data to be downloaded.
    :param end_date: string representing the end date of the data to be downloaded.
    :return: pandas DataFrame containing the yfinance data of the securities.
    """
    return yf.download(tickers, start=start_date, end=end_date, interval=interval)


if __name__ == '__main__':
    # Testing the download_ticker_data function
    tickers = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'META']
    start_date = '2018-01-01'
    end_date = '2019-12-31'
    data = download_ticker_data(tickers, start_date, end_date)
    print(data)
