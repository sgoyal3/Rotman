import requests
import pandas as pd
import numpy as np
API_KEY = {'X-API-key': 'LBFAK2UH'}
port = "9999"


#submit orders
# http://localhost:8080/v1/orders&ticker=CRZY&type=MARKET&quantity=100
#&action=BUY, where &ticker=CRZY&type=MARKET&quantity=100&action=BUY
with requests.Session() as s:
    s.headers.update(API_KEY)

def get_time_remaining():
    s.headers.update(API_KEY)
    resp = s.get('http://localhost:{}/v1/case'.format(port))
    if resp.status_code == 200:
        case = resp.json()
        tick = case['tick']
        return 300 - tick
    return 0


def get_tender_info():
    s.headers.update(API_KEY)
    resp = s.get('http://localhost:{}/v1/tenders'.format(port))
    if resp.status_code == 200:
        tender = resp.json()
        if len(tender) > 0:
            return tender[0]['tender_id'], tender[0]['quantity'], tender[0]['action'], tender[0]['price']
    return

def get_bid_and_ask(ticker):
    s.headers.update(API_KEY)
    resp = s.get('http://localhost:{}/v1/securities/book?ticker={}&limit=1'.format(port, ticker))
    if resp.status_code == 200:
        order = resp.json()
        return order['bids'][0]['price'], order['bids'][0]['quantity'], order['asks'][0]['price'], order['asks'][0]['quantity']
    return

def add_market_order(ticker, vol, sell):
    s.headers.update(API_KEY)
    if sell:
        action = 'SELL'
    else:
        action = 'BUY'
    params = {'ticker': ticker, 'type': 'MARKET', 'quantity': vol, 'action': action}
    resp = s.post('http://localhost:{}/v1/orders'.format(port), params= params)
    if resp.status_code == 200:
        print("Successfully submitted market order")
    return

def add_limit_order(ticker, vol, price, sell):
    if sell:
        action = 'SELL'
    else:
        action = 'BUY'
    params = {'ticker': ticker, 'type': 'LIMIT', 'quantity': vol,
              'action': action, 'price': price}
    resp = s.post('http://localhost:{}/v1/orders'.format(port), params= params)
    if resp.status_code == 200:
        print("Added limit order to ", action, " ", vol, " ", ticker)
    return

def vol_imbalance(bid_vol, ask_vol):
    return (bid_vol - ask_vol)/(bid_vol + ask_vol)

def flow_imbalance(bid_vol, ask_vol, lag_bid_vol, lag_ask_vol, bid_price, ask_price, lag_bid_price, lag_ask_price):
    if bid_price < lag_bid_price:
        deltaB = 0
    elif bid_price == lag_bid_price:
        deltaB = bid_vol - lag_bid_vol
    else:
        deltaB = bid_vol
    if ask_price < lag_ask_price:
        deltaA = ask_vol
    elif ask_price == lag_ask_price:
        deltaA = ask_vol - lag_ask_vol
    else:
        deltaA = 0
    return deltaB - deltaA

def get_ticker_info(ticker, info):
    resp = s.get('http://localhost:{}/v1/securities?ticker={}'.format(port, ticker))
    if resp.status_code == 200:
        security = resp.json()
        if info == 'position':
            return security[0]['position']
    return 0

def market_unload(ticker, volume):
    if volume > 0:
        while volume > 10000:
            add_market_order(ticker, 10000,sell = True)
            volume = volume - 10000
        add_market_order(ticker, volume, sell = True)
    else:
        while volume < -10000:
            add_market_order(ticker, 10000, sell = False)
            volume = volume + 10000
        add_market_order(ticker, (-1)*volume, sell = False)
    return

def tender_accept():
    tender = get_tender_info()
    if tender:
        resp = s.post('http://localhost:{}/v1/tenders/{}'.format(port, str(tender[0])))
        if resp.status_code == 200:
            print("accepted tender!")
    return


class AggOrder:
    def __init__(self, order_json, marketOrder=True):
        self.init = False
        self.price = None
        self.action = None
        self.type = None
        self.volume_unfilled = self.volume_filled = None
        self.id = None
        # indicate whether this is order object in market book
        self.marketOrder = marketOrder
        if marketOrder:
            self.aggregate_market(order_json)
        else:
            self.aggregate_trader(order_json)

    def __repr__(self):
        output = "" if self.marketOrder else "id: {} ".format(self.id)
        output += "Order({}, {}, price: {}, volume_unfilled: {})".format(
            self.action, self.type, self.price, self.volume_unfilled)
        return output

    def aggregate_market(self, order_json):
        # need to aggregate
        if self.init == False:
            self.init = True
            self.price = order_json["price"]
            self.type = order_json["type"]
            self.action = order_json["action"]
            self.volume_filled = order_json["quantity_filled"]
            self.volume_unfilled = order_json["quantity"] - order_json["quantity_filled"]
        else:
            self.volume_filled += order_json["quantity_filled"]
            self.volume_unfilled += (order_json["quantity"] - order_json["quantity_filled"])

    # don't have to call this for trader book
    def aggregate_trader(self, order_json):
        if self.init == False:
            self.init = True
            self.id = [order_json["order_id"]]
            self.price = order_json["price"]
            self.type = order_json["type"]
            self.action = order_json["action"]
            self.volume_filled = order_json["quantity_filled"]
            self.volume_unfilled = order_json["quantity"] - order_json["quantity_filled"]
        else:
            self.id.append(order_json["order_id"])
            self.volume_filled += order_json["quantity_filled"]
            self.volume_unfilled += (order_json["quantity"] - order_json["quantity_filled"])


class OrderBook:
    def __init__(self, session, size=5):
        self.s = session
        self.size = size
        # limit of queries to send
        self.record_limit = 3 * size
        # limit of number of entries maintained in the book
        self.marketBookSize = size
        self.tickers = ["BULL", "BEAR", "RITC"]
        self.actions = ["bids", "asks"]

        # current and lag order book
        self.marketBook = self.create_new_book()
        self.traderBook = self.create_new_book()
        self.lag_marketBook = self.create_new_book()

        # threshold for keeping order entry in the book
        self.spread_threshold = .1

        # trailing vwap used for strategy
        self.trailing_vwap = []

        # get trader id for filtering orders
        resp = self.s.get('http://localhost:{}/v1/trafer')
        if resp.status_code == 200:
            self.trader_id = resp.json()["trader_id"]

    def create_new_book(self):
        output = dict.fromkeys(self.tickers)
        for k in self.tickers:
            output[k] = dict.fromkeys(self.actions)
        return output

    # market orders are arranged in price (logically reasonable for Bid and Ask)
    # this only updates order info for the specified ticker!
    def updateMarketBook(self, ticker):
        if ticker not in self.tickers:
            print("Check Ticker!")
            return
        resp = s.get('http://localhost:{}/v1/securities/book?ticker={}&limit={}'.format(
            port, ticker, self.record_limit))
        if resp.status_code == 200:
            # update lag_order book and clear out prev records
            self.lag_marketBook[ticker] = self.marketBook[ticker]
            self.marketBook[ticker] = dict.fromkeys(self.actions)

            content = resp.json()
            for order_type in self.actions:
                if len(content[order_type]) == 0:
                    continue
                # maintain a list of length self.size
                self.marketBook[ticker][order_type] = []
                prev_price = None

                # price we maintain in order book
                price_arr = list(dict.fromkeys([item["price"] for item in content[order_type]]))[:self.size]
                bchmk_price = price_arr[-1]

                # check whether to keep the last few entries in the book
                # kinda arbitrary...
                num_items_check = 2 if self.marketBookSize <= 6 else 4
                for p in price_arr[::-1]:
                    if np.abs(p - bchmk_price) > self.spread_threshold:
                        price_arr = price_arr[:-1]

                for order in content[order_type]:
                    # filter out our orders
                    if order['trader_id'] == self.trader_id:
                        continue

                    if order['action'] == "BUY" and order['price'] < bchmk_price or \
                            order['action'] == "SELL" and order['price'] > bchmk_price:
                        break

                    if order['price'] != prev_price:
                        prev_price = order['price']
                        self.marketBook[ticker][order_type].append(AggOrder(order))
                    else:
                        self.marketBook[ticker][order_type][-1].aggregate_market(order)
        else:
            print(resp.json())

    def updateAllMarketBook(self):
        for k in self.tickers:
            self.updateMarketBook(k)

    # Trader Book can be arraged in differnt orders!
    # if order by time, then in descending order of time that orders stayed open
    # if order by price, then in logical order of price (bid vs ask)
    def updateTraderBook(self, ticker, orderBy="time", aggregate=False):
        if ticker not in self.tickers:
            print("Check Ticker!")
            return
        resp = s.get('http://localhost:{}/v1/orders?status=OPEN'.format(port))

        if resp.status_code == 200:
            # clear out prev records
            self.traderBook[ticker] = dict.fromkeys(self.actions)
            self.traderBook[ticker]["bids"] = []
            self.traderBook[ticker]["asks"] = []
            content = resp.json()
            if len(content) == 0:
                return
            # keep track of individual open order
            if not aggregate:
                for order in content:
                    if order['action'] == "BUY":
                        self.traderBook[order["ticker"]]["bids"].append(AggOrder(order, marketOrder=False))
                    else:
                        self.traderBook[order["ticker"]]["asks"].append(AggOrder(order, marketOrder=False))
            else:
                prev_price = None
                for order in content:
                    if order['price'] != prev_price:
                        prev_price = order['price']
                        self.traderBook[ticker][order_type].append(AggOrder(order, marketOrder=False))
                    else:
                        self.traderBook[ticker][order_type][-1].aggregate_trader(order)

            if orderBy == "price":
                self.traderBook[ticker]["bids"].sort(key=lambda x: -x.price)
                self.traderBook[ticker]["asks"].sort(key=lambda x: x.price)

        else:
            print(resp.json())

    def get_market_volume_and_vwap(self, ticker):
        vwap_bid = vwap_ask = 0.0
        volume_bid = volume_ask = 0.0

        for order in self.marketBook[ticker]["bids"]:
            vwap_bid += order.volume_unfilled * order.price
            volume_bid += order.volume_unfilled
        vwap_bid /= volume_bid

        for order in self.marketBook[ticker]["asks"]:
            vwap_ask += order.volume_unfilled * order.price
            volume_ask += order.volume_unfilled
        vwap_ask /= volume_ask

        return volume_bid, volume_ask, vwap_bid, vwap_ask

    def print_market_book(self, ticker):
        print("Bid: {}".format(self.marketBook[ticker]["bids"]))
        print("Ask: {}".format(self.marketBook[ticker]["asks"]))

    def print_trader_book(self, ticker):
        print("Bid: {}".format(self.traderBook[ticker]["bids"]))
        print("Ask: {}".format(self.traderBook[ticker]["asks"]))

    def check_data_available(self, ticker):
        try:
            curr_avail = len(self.marketBook[ticker]["bids"]) and len(self.marketBook[ticker]["asks"])
            lag_avail = len(self.lag_marketBook[ticker]["bids"]) and len(self.lag_marketBook[ticker]["asks"])
        except TypeError:
            return False, False
        return bool(lag_avail), bool(curr_avail)

    # Stats for Spread:
    def get_bid_ask(self, ticker):
        return self.marketBook[ticker]["bids"][0].price, self.marketBook[ticker]["asks"][0].price

    def get_bid_ask_spread(self, ticker):
        return self.marketBook[ticker]["asks"][0].price - self.marketBook[ticker]["bids"][0].price

    def get_mid_price(self, ticker, vwap=False):
        if not vwap:
            return (self.marketBook[ticker]["asks"][0].price + self.marketBook[ticker]["bids"][0].price) / 2.0
        else:
            _, _, vwap_bid, vwap_ask = self.get_market_volume_and_vwap(ticker)
            return (vwap_bid + vwap_ask) / 2.0

    # if use vwap = False, then just the best bid/ask price
    def get_vol_imbalance(self, ticker, vwap=False):
        if not vwap:
            bid_vol = self.marketBook[ticker]["bids"][0].volume_unfilled
            ask_vol = self.marketBook[ticker]["asks"][0].volume_unfilled
        else:
            bid_vol, ask_vol, _, _ = self.get_market_volume_and_vwap(ticker)

        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def get_flow_imbalance(self, ticker):
        best_bid = self.marketBook[ticker]["bids"][0]
        best_ask = self.marketBook[ticker]["asks"][0]
        lag_best_bid = self.lag_marketBook[ticker]["bids"][0]
        lag_best_ask = self.lag_marketBook[ticker]["asks"][0]
        deltaA = deltaB = None

        if best_bid.price < lag_best_bid.price:
            deltaB = 0
        elif best_bid.price == lag_best_bid.price:
            deltaB = best_bid.volume_unfilled - lag_best_bid.volume_unfilled
        else:
            deltaB = best_bid.volume_unfilled

        if best_ask.price < lag_best_ask.price:
            deltaA = best_ask.volume_unfilled
        elif best_ask.price == lag_best_ask.price:
            deltaA = best_ask.volume_unfilled < lag_best_ask.volume_unfilled
        else:
            deltaA = 0
        return deltaB - deltaA

    # trading api
    def cancel_bad_orders(self, ticker, action, spread_threshold=0.0):
        try:
            cancel_ids = []
            best_price = self.marketBook[ticker][action][0].price
            for order in self.traderBook[ticker][action]:
                if (action == "bids" and best_price - order.price <= spread_threshold) or \
                        (action == "asks" and order.price - best_price <= spread_threshold):
                    cancel_ids.extend(order.id)
            if len(cancel_ids):
                cancel_ids = [str(id) for id in cancel_ids]
                resp = s.post('http://localhost:{}/v1/commands/cancel?ids={}'.format(port, ','.join(cancel_ids)))
                if resp.status_code == 200:
                    print("Canceled orders: {}".format(cancel_ids))

        except TypeError:
            return


def main():
    myOrderBook = OrderBook(s, size=5)

    while get_time_remaining() > 1:

        myOrderBook.updateAllMarketBook()
        myOrderBook.updateTraderBook("RITC", orderBy="price")

        lag_data_avail, curr_data_avail = myOrderBook.check_data_available("RITC")

        VB = FB = 0

        if curr_data_avail:
            VB = myOrderBook.get_vol_imbalance("RITC")
            if lag_data_avail:
                FB = myOrderBook.get_flow_imbalance("RITC")

        current_bid, current_ask = myOrderBook.get_bid_ask("RITC")

        if VB > 0.3:
            add_limit_order('RITC', 1000, current_bid + 0.01, None)
            myOrderBook.cancel_bad_orders("RITC", "asks", spread_threshold=0.01)
            add_limit_order('RITC', 1000, current_ask, 1)
        if VB < -0.3:
            add_limit_order('RITC', 1000, current_ask - 0.01, 1)
            myOrderBook.cancel_bad_orders("RITC", "bids", spread_threshold=0.01)
            add_limit_order('RITC', 1000, current_bid, None)

        myOrderBook.updateTraderBook("RITC", orderBy="price")

        if FB > 5000:
            add_limit_order('RITC', 1000, current_bid + 0.01, None)
            myOrderBook.cancel_bad_orders("RITC", "asks", spread_threshold=0.01)
            add_limit_order('RITC', 1000, current_ask, 1)

        if FB < -5000:
            add_limit_order('RITC', 1000, current_ask - 0.01, 1)
            myOrderBook.cancel_bad_orders("RITC", "bids", spread_threshold=0.01)
            add_limit_order('RITC', 1000, current_bid, None)

        if np.random.uniform(0, 1, 1) > 0.8:
            resp = s.post('http://localhost:{}/v1/commands/cancel?all=1'.format(port))

        position = get_ticker_info('RITC', 'position')

        if current_ask - current_bid > 0.04:
            add_limit_order('RITC', 500, current_bid, None)
            add_limit_order('RITC', 500, current_bid - 0.01, None)
            add_limit_order('RITC', 500, current_bid - 0.02, None)
            add_limit_order('RITC', 500, current_ask - 0.01, 1)
            add_limit_order('RITC', 500, current_ask, 1)
            add_limit_order('RITC', 500, current_ask + 0.01, 1)
            add_limit_order('RITC', 500, current_ask + 0.02, 1)

        if position <= -1000:
            add_limit_order('RITC', 1000, current_bid + 0.01, None)
            add_limit_order('RITC', 1000, current_bid, None)
            add_limit_order('RITC', 1000, current_bid - 0.01, None)
            add_limit_order('RITC', 1000, current_bid - 0.02, None)
            add_limit_order('RITC', 1000, current_bid - 0.03, None)
            add_limit_order('RITC', 1000, current_bid - 0.04, None)

        if position >= 1000:
            add_limit_order('RITC', 1000, current_ask - 0.01, 1)
            add_limit_order('RITC', 1000, current_ask, 1)
            add_limit_order('RITC', 1000, current_ask + 0.01, 1)
            add_limit_order('RITC', 1000, current_ask + 0.02, 1)
            add_limit_order('RITC', 1000, current_ask + 0.03, 1)
            add_limit_order('RITC', 1000, current_ask + 0.04, 1)


if __name__ == '__main__':
    main()