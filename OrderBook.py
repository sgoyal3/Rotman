import numpy as np
import requests
import logging


def check_case_status():
    resp = SESSION.get('http://localhost:{}/v1/case'.format(PORT))
    if resp.status_code == 200:
        status = resp.json()["status"]
        if status == "ACTIVE":
            return True
    return False


def get_tick():
    resp = SESSION.get('http://localhost:{}/v1/case'.format(PORT))
    if resp.status_code == 200:
        return resp.json()['tick']


def get_trader_id():
    # get trader id for filtering orders
    resp = SESSION.get('http://localhost:{}/v1/trader'.format(PORT))
    if resp.status_code == 200:
        trader_id = resp.json()["trader_id"]
        return trader_id
    return None


def add_market_order(ticker, vol, sell):
    SESSION.headers.update(API_KEY)
    action = "SELL" if sell else "BUY"
    params = {'ticker': ticker, 'type': 'MARKET', 'quantity': vol,
              'action': action}
    resp = SESSION.post('http://localhost:{}/v1/orders'.format(PORT),
                        params=params)
    if resp.status_code == 200:
        print("Successfully submitted market order")
    return


def add_limit_order(ticker, vol, price, sell):
    action = "SELL" if sell else "BUY"
    params = {'ticker': ticker, 'type': 'LIMIT', 'quantity': vol,
              'action': action, 'price': price}
    resp = SESSION.post('http://localhost:{}/v1/orders'.format(PORT),
                        params=params)
    if resp.status_code == 200:
        print("Added limit order to ", action, " ", vol, " ", ticker)
    return


# split the order into num_splits, the first order starts at best_price
def split_order(ticker, best_price, quantity, sell, num_splits=4,
                price_spread=.01):
    quantities = [int(quantity / num_splits)] * (num_splits - 1)
    quantities.append(quantity - sum(quantities))
    if not sell:
        prices = [best_price - price_spread * i for i in range(num_splits)]
        for i in range(num_splits):
            add_limit_order(ticker, quantities[i], prices[i], False)
    else:
        prices = [best_price + price_spread * i for i in range(num_splits)]
        for i in range(num_splits):
            add_limit_order(ticker, quantities[i], prices[i], True)


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
        if not self.init:
            self.init = True
            self.price = order_json["price"]
            self.type = order_json["type"]
            self.action = order_json["action"]
            self.volume_filled = order_json["quantity_filled"]
            self.volume_unfilled = order_json["quantity"] - \
                order_json["quantity_filled"]
        else:
            self.volume_filled += order_json["quantity_filled"]
            self.volume_unfilled += (order_json["quantity"] -
                                     order_json["quantity_filled"])

    # don't have to call this for trader book
    def aggregate_trader(self, order_json):
        if not self.init:
            self.init = True
            self.id = [order_json["order_id"]]
            self.price = order_json["price"]
            self.type = order_json["type"]
            self.action = order_json["action"]
            self.volume_filled = order_json["quantity_filled"]
            self.volume_unfilled = order_json["quantity"] - \
                order_json["quantity_filled"]
        else:
            self.id.append(order_json["order_id"])
            self.volume_filled += order_json["quantity_filled"]
            self.volume_unfilled += (order_json["quantity"] -
                                     order_json["quantity_filled"])


# manage the market order book
class OrderBook:
    def __init__(self, bookType, size=5):
        # limit of queries to send
        self.record_limit = 3 * size
        # limit of number of entries maintained in the book
        self.bookSize = size

        self.bookTypeSet = ["MARKET", "TRADER"]
        assert bookType in self.bookTypeSet
        self.bookType = bookType
        self.tickers = ["BULL", "BEAR", "USD", "RITC"]
        # to be consistent with the json format, use BUY instead of Bid
        self.actions = ["BUY", "SELL"]

        self.orderBook = self.create_new_book()

        # threshold for keeping order entry in the book
        # if too far from the best price, don't include
        self.spread_threshold = .05

        # trailing data useful for strategy
        self.volume_bid_arr = []
        self.volume_ask_arr = []
        self.vwap_bid_arr = []
        self.vwap_ask_arr = []
        self.best_bid_arr = []
        self.best_ask_arr = []
        # keep track of how many times the order book has been updated
        # (can be multiple times in a single tick)
        self.RITC_record_count = 0
        self.trader_id = None if self.bookType == "TRADER" else get_trader_id()

    # utility functions
    def check_data_available(self):
        return self.RITC_record_count > 0

    def create_new_book(self):
        output = dict.fromkeys(self.tickers)
        for k in self.tickers:
            output[k] = dict.fromkeys(self.actions)
        return output

    def print_order_book(self, ticker):
        print("Bid: {}".format(self.orderBook[ticker]["BUY"]))
        print("Ask: {}".format(self.orderBook[ticker]["SELL"]))

    # data updating functions
    # market orders are arranged in price (logically reasonable for Bid and Ask)
    # this only updates order info for the specified ticker!
    def updateMarketBook(self, ticker):
        assert self.bookType == "MARKET"
        assert ticker in self.tickers

        resp = SESSION.get(
            'http://localhost:{}/v1/securities/book?ticker={}&limit={}'.format(
                PORT, ticker, self.record_limit))

        if resp.status_code == 200:
            # create a new dict
            self.orderBook[ticker] = dict.fromkeys(self.actions)
            content = resp.json()
            # check BUY / SELL
            for order_type in self.actions:
                json_order_type = "bids" if order_type == "BUY" else "asks"
                if len(content[json_order_type]) == 0:
                    continue
                # maintain a list of length self.size
                self.orderBook[ticker][order_type] = []
                prev_price = None

                # price we maintain in order book (the first min(book_size, entry_size) entries)
                # sorted in logical order!
                price_arr = list(dict.fromkeys([item["price"] for item in content[json_order_type]]))[
                    :self.bookSize]
                best_price = price_arr[0]

                # check whether to keep the last few entries in the book
                # kinda arbitrary...
                price_arr = [p for p in price_arr if np.abs(
                    p - best_price) < self.spread_threshold]
                worst_price = price_arr[-1]

                for order in content[json_order_type]:
                    # filter out our orders
                    if order['trader_id'] == self.trader_id:
                        continue
                    # exclude orders with worse prices
                    if order['action'] == "BUY" and order['price'] < worst_price or \
                            order['action'] == "SELL" and order['price'] > worst_price:
                        break
                    # aggregate orders
                    if order['price'] != prev_price:
                        prev_price = order['price']
                        self.orderBook[ticker][order_type].append(
                            AggOrder(order))
                    else:
                        self.orderBook[ticker][order_type][-1].aggregate_market(
                            order)

            # automatically update the trailing history if RITC
            if ticker == "RITC":
                self.RITC_record_count += 1
                self.update_trailing_data_RITC()
        else:
            print(resp.json())

    def updateAllMarketBook(self):
        assert self.bookType == "MARKET"
        for k in self.tickers:
            self.updateMarketBook(k)

    # Trader Book can be arranged in different orders!
    # if order by time, then in descending order of time that orders submitted
    # if order by price, then in logical order of price (bid vs ask)
    def updateTraderBook(self, ticker, orderBy="time", aggregate=False):
        assert self.bookType == "TRADER"
        assert ticker in self.tickers

        resp = SESSION.get(
            'http://localhost:{}/v1/orders?status=OPEN'.format(PORT))
        if resp.status_code == 200:
            # clear out prev records
            self.orderBook[ticker] = dict.fromkeys(self.actions)
            self.orderBook[ticker]["BUY"] = []
            self.orderBook[ticker]["SELL"] = []
            content = resp.json()
            # if open orders
            if len(content) == 0:
                return
            # keep track of individual open order
            if not aggregate:
                for order in content:
                    if order['action'] == "BUY":
                        self.orderBook[order["ticker"]]["BUY"].append(
                            AggOrder(order, marketOrder=False))
                    else:
                        self.orderBook[order["ticker"]]["SELL"].append(
                            AggOrder(order, marketOrder=False))
            else:
                for order_type in self.actions:
                    prev_price = None
                    for order in content:
                        if order['price'] != prev_price:
                            prev_price = order['price']
                            self.orderBook[ticker][order_type].append(
                                AggOrder(order, marketOrder=False))
                        else:
                            self.orderBook[ticker][order_type][
                                -1].aggregate_trader(
                                order)
            if orderBy == "price":
                self.orderBook[ticker]["BUY"].sort(key=lambda x: -x.price)
                self.orderBook[ticker]["SELL"].sort(key=lambda x: x.price)
        else:
            print(resp.json())

    def update_trailing_data_RITC(self):
        assert self.bookType == "MARKET"

        if self.RITC_record_count > 0:
            vwap_bid = vwap_ask = 0.0
            volume_bid = volume_ask = 0.0

            for order in self.orderBook["RITC"]["BUY"]:
                vwap_bid += order.volume_unfilled * order.price
                volume_bid += order.volume_unfilled

            for order in self.orderBook["RITC"]["SELL"]:
                vwap_ask += order.volume_unfilled * order.price
                volume_ask += order.volume_unfilled

            vwap_ask /= volume_ask
            vwap_bid /= volume_bid

            try:
                self.best_bid_arr.append(self.orderBook["RITC"]["BUY"][0].price)
                self.best_ask_arr.append(self.orderBook["RITC"]["SELL"][0].price)
                self.volume_bid_arr.append(volume_bid)
                self.volume_ask_arr.append(volume_ask)
                self.vwap_bid_arr.append(vwap_bid)
                self.vwap_ask_arr.append(vwap_ask)
            except IndexError:
                return

    # Statistics functions

    # get latest vwap and volume
    # same code, but can call on other tickers
    def get_market_volume_and_vwap(self, ticker):
        assert self.bookType == "MARKET"

        vwap_bid = vwap_ask = 0.0
        volume_bid = volume_ask = 0.0

        for order in self.orderBook[ticker]["BUY"]:
            vwap_bid += order.volume_unfilled * order.price
            volume_bid += order.volume_unfilled
        vwap_bid /= volume_bid

        for order in self.orderBook[ticker]["SELL"]:
            vwap_ask += order.volume_unfilled * order.price
            volume_ask += order.volume_unfilled
        vwap_ask /= volume_ask
        return volume_bid, volume_ask, vwap_bid, vwap_ask

    # for filtering trader order book
    def get_trader_volume(self, ticker, action, price_threshold=0.05):
        assert self.bookType == "TRADER"

        volume = 0
        bchmk_price = self.orderBook[ticker][action][0].price

        for order in self.orderBook[ticker][action]:
            if np.abs(order.price - bchmk_price) > price_threshold:
                break
            volume += order.volume_unfilled
        return volume

    def get_RITC_trailing_stat(self, hist_len=20):
        assert self.bookType == "MARKET"
        if len(self.volume_bid_arr) < hist_len:
            return None
        output = {}
        vwap_mid = (np.array(self.vwap_bid_arr[-hist_len:]) + np.array(
            self.vwap_ask_arr[-hist_len:])) / 2.0
        bid_ask_mid = (np.array(self.best_bid_arr[-hist_len:]) + np.array(
            self.best_ask_arr[-hist_len:])) / 2.0
        output["vwap_mid_avg"] = np.mean(vwap_mid)
        output["vwap_mid_vol"] = np.std(vwap_mid)
        output["mid_price_avg"] = np.mean(bid_ask_mid)
        output["mid_price_vol"] = np.std(bid_ask_mid)
        return output

    def get_volume(self, ticker, sell):
        side = "SELL" if sell else "BUY"
        return sum(order.volume_unfilled for order
                   in self.orderBook[ticker][side])

    def get_usd(self, sell):
        action = "SELL" if sell else "BUY"
        return self.orderBook["USD"][action][0].price

    def get_best_price(self, ticker, sell):
        action = "SELL" if sell else "BUY"
        return self.orderBook[ticker][action][0].price

    # Stats for Spread:
    def get_bid_ask_and_spread(self, ticker):
        bid = self.orderBook[ticker]["BUY"][0].price
        ask = self.orderBook[ticker]["SELL"][0].price
        return bid, ask, ask - bid

    def get_mid_price(self, ticker, vwap=False):
        if not vwap:
            return (self.orderBook[ticker]["SELL"][0].price +
                    self.orderBook[ticker]["BUY"][0].price) / 2.0
        else:
            _, _, vwap_bid, vwap_ask = self.get_market_volume_and_vwap(ticker)
            return (vwap_bid + vwap_ask) / 2.0

    def get_vol_imbalance(self):
        bid_vol = self.volume_bid_arr[-1]
        ask_vol = self.volume_ask_arr[-1]
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def get_flow_imbalance(self):
        lag_best_bid, best_bid = self.best_bid_arr[-2:]
        lag_best_ask, best_ask = self.best_ask_arr[-2:]
        lag_best_bid_vol, best_bid_vol = self.volume_bid_arr[-2:]
        lag_best_ask_vol, best_ask_vol = self.volume_ask_arr[-2:]

        deltaA = deltaB = None
        if best_bid < lag_best_bid:
            deltaB = 0
        elif best_bid == lag_best_bid:
            deltaB = best_bid_vol - lag_best_bid_vol
        else:
            deltaB = best_bid_vol

        if best_ask < lag_best_ask:
            deltaA = best_ask_vol
        elif best_ask == lag_best_ask:
            deltaA = best_ask_vol - lag_best_ask_vol
        else:
            deltaA = 0
        return deltaB - deltaA

    def hedge_usd(self, size, sell):
        usd_best = self.get_best_price("USD", sell)
        add_limit_order("USD", size, usd_best, sell)


# manage our own book and size the trade
class PositionManager:
    def __init__(self):
        self.positionBook = dict.fromkeys(["RITC", "BEAR", "BULL"])
        # limit for RITC (1 ritc ~ 2 stocks)
        # limit: 120K if have tender offers else 50K
        self.tender_position_limit = 100000
        self.trade_position_limit = 50000
        self.ideal_position_limit = 25000
        self.low_position_limit = 10000
        self.RITC_order_limit = 10000
        self.need_unwind = False
        self.bchmk_price = None

    def get_position(self, ticker):
        return self.positionBook[ticker]["position"]

    # return the upper bound for trading
    def market_sizing(self, sell):
        max_book_position = 0.1
        # if have over 50K net position in ritc, then don't trade more
        # ideally, max 10% of the liquidity
        # in max case, have 50K after trading
        if np.abs(self.positionBook["RITC"]["position"]) > self.trade_position_limit:
            return 0
        elif np.abs(self.positionBook["RITC"]["position"]) > self.trade_position_limit / 2:
            temp = int(max_book_position / 2 *
                       MARKETBOOK.get_volume("RITC", sell))
            return min(temp, self.trade_position_limit - np.abs(MARKETBOOK.get_volume("RITC", sell)))
        else:
            temp = int(max_book_position * MARKETBOOK.get_volume("RITC", sell))
            return min(temp, self.trade_position_limit - np.abs(MARKETBOOK.get_volume("RITC", sell)))

    def check_need_unwind(self):
        temp = np.abs(self.positionBook["RITC"]["position"])
        if temp > self.ideal_position_limit:
            self.need_unwind = True
        else:
            self.need_unwind = False
        return self.need_unwind

    # logic:
    # 1. if liquidity is good, just unwind in limit order (check benchmark price)
    # 2. else, if arbitrage exists, do it
    # 3. else, if converter is
    def unwind_position(self, sell):
        size = self.positionBook["RITC"]["position"]
        # need to have updated price for Bull / Bear
        MARKETBOOK.updateAllMarketBook()

        while np.abs(size) > self.trade_position_limit:
            if not self.try_arbitrage(size, sell=sell):
                if not self.try_converter(sell=sell):
                    order_size = min(self.market_sizing(sell=sell),
                                     self.RITC_order_limit, np.abs(size / 3))
                    self.add_split_orders("RITC", num_splits=3, sell=sell, order_size=order_size)
            self.updatePositionBook("RITC")
            size = self.positionBook["RITC"]["position"]

        # cancel orders
        POSITION_MANAGER.cancel_bad_orders(
            "RITC", sell=sell, close_to_best=True, spread_threshold=.05)

    # no while loop...
    # attempt arbitrage function, only executes if both arbitrage is available and it helps us unwind position in RITC
    # sell = True if want to sell RITC
    def try_arbitrage(self, size, sell):
        # While max RITC to buy/sell for arbitrage is greater than certain threshold
        USD = MARKETBOOK.get_usd(sell)
        # check price / volume on opposite side
        bull_vol = MARKETBOOK.get_volume("BULL", sell)
        bear_vol = MARKETBOOK.get_volume("BEAR", sell)
        RITC_vol = MARKETBOOK.get_volume("RITC", not sell)
        limit = min(bull_vol * 2, bear_vol * 2, RITC_vol, size)

        RITC = MARKETBOOK.get_best_price("RITC", not sell)
        Bull = MARKETBOOK.get_best_price("BULL", sell)
        Bear = MARKETBOOK.get_best_price("BEAR", sell)
        price_check = (sell and (RITC - 0.01) * USD > Bull + Bear) or \
                      (not sell and (RITC + 0.01) * USD < Bull + Bear)

        if price_check:
            # Cancel all outstanding orders
            POSITION_MANAGER.cancel_bad_orders(
                "RITC", not sell, close_to_best=True, spread_threshold=.03)
            # Buy RITC, sell Bull, Bear, buy USD
            add_market_order("RITC", 0.7 * limit, sell)
            add_market_order("BULL", 0.7 * limit, not sell)
            add_market_order("BEAR", 0.7 * limit, not sell)
            # hedging
            add_market_order("USD", 0.7 * limit * (Bear + Bull), sell)
            return True
        return False

    def try_converter(self, sell):
        # no hedging here....
        USD = MARKETBOOK.get_usd(sell)
        fixed_cost = 1500
        bull = MARKETBOOK.get_mid_price("BULL", vwap=True)
        bear = MARKETBOOK.get_mid_price("BEAR", vwap=True)
        vwap_mid = MARKETBOOK.get_mid_price("RITC", vwap=True)

        if not sell:
            # positive RITC position, negative bull / bear
            price_spread = (vwap_mid * USD - bull - bear) * \
                POSITION_MANAGER.low_position_limit - fixed_cost
            if price_spread > 0:
                print(
                    "\nKurtis / Jason, use converter from {} to {}!\n".format("Bull, Bear", "RITC"))
                return True
        else:
            price_spread = (-vwap_mid * USD + bull + bear) * \
                POSITION_MANAGER.low_position_limit - fixed_cost
            if price_spread > 0:
                print(
                    "\nKurtis / Jason, use converter from {} to {}!\n".format("RITC", "Bull, Bear"))
                return True
        return False

    def updatePositionBook(self, ticker):
        resp = SESSION.get(
            'http://localhost:{}/v1/securities?ticker={}'.format(PORT, ticker))
        if resp.status_code == 200:
            # clear out prev records
            self.positionBook[ticker] = resp.json()[0]

    # add orders if have signals, not to unwind position
    def add_split_orders(self, ticker, num_splits, sell, order_size=None):
        curr_position = self.positionBook[ticker]["position"]
        if not sell:
            if order_size is None:
                total_size = 0 if curr_position > self.trade_position_limit else 1000 \
                    if curr_position > self.ideal_position_limit else 2000
            else:
                total_size = order_size
            split_order(ticker, MARKETBOOK.best_bid_arr[-1],
                        total_size, sell=False, num_splits=num_splits)
            MARKETBOOK.hedge_usd(total_size, sell=False)
        else:
            if order_size is None:
                total_size = 0 if curr_position < -self.trade_position_limit else 1000 \
                    if curr_position < -self.ideal_position_limit else 2000
            else:
                total_size = order_size
            split_order(ticker, MARKETBOOK.best_ask_arr[-1],
                        total_size, sell=True, num_splits=num_splits)
            MARKETBOOK.hedge_usd(total_size, sell=True)

    # Trading functions
    # if close_to_best = True, then cancel orders close to the best bid/ask
    def cancel_bad_orders(self, ticker, sell, close_to_best=True, spread_threshold=0.0):
        action = "SELL" if sell else "BUY"
        try:
            cancel_ids = []
            best_price = MARKETBOOK.get_best_price(ticker, sell=sell)
            for order in TRADERBOOK.orderBook[ticker][action]:
                if close_to_best:
                    if (action == "BUY" and best_price - order.price <= spread_threshold) or \
                            (action == "SELL" and order.price - best_price <= spread_threshold):
                        cancel_ids.extend(order.id)
                else:
                    if (action == "BUY" and best_price - order.price > spread_threshold) or \
                            (action == "SELL" and order.price - best_price > spread_threshold):
                        cancel_ids.extend(order.id)
            if len(cancel_ids):
                cancel_ids = [str(id) for id in cancel_ids]
                resp = SESSION.post(
                    'http://localhost:{}/v1/commands/cancel?ids={}'.format(
                        PORT, ','.join(cancel_ids)))
                if resp.status_code == 200:
                    print("Canceled orders: {}".format(cancel_ids))
        except TypeError:
            return


class TenderManager:
    class tenderOffer:
        def __init__(self, tender_json):
            self.tender_id = tender_json[0]['tender_id']
            self.quantity = tender_json[0]['quantity']
            self.action = tender_json[0]['action']
            self.price = tender_json[0]['price']

        def __repr__(self):
            return "id: {}, action: {}, price: {}, quantity: {}".format(
                self.tender_id, self.action, self.price, self.quantity)

    def __init__(self, wait_tick=3):
        # only these two states persist
        self.state_set = {"NOOFFER", "FRONTRUN"}
        self.state = "NOOFFER"
        # keep track of received offers
        self.last_tick_took = None
        self.wait_tick = wait_tick
        self.offer = None

        # TODO: set critical params as Global Vars
        # parameters for evaluation
        self.volatility_mult_before = 2
        self.volatility_mult_after = 1
        self.volume_threshold = .8

    # Tender Offer Logic:
    # 1. If receive new offers, evaluate: offer price vs The current vwap +
    # vwap_trailing_std + reasonable spread, if profitable, decide to take it
    # 2. If have decided to take the offer, front run by submitting half of the
    # size of the order on the opposite direction and wait until 4 ticks later
    # 3. After 4 tickers, then evaluate again whether it's still profitable
    # (in terms of vwap + trailing_std + spread,
    # or how our previous orders on opposite side were filled)
    # 4. If market goes against us, decline the offer, keep unfilled orders
    # on the opposite side open (???)
    def handle_offer(self):
        # check if have new offers
        if self.state == "NOOFFER":
            resp = SESSION.get('http://localhost:{}/v1/tenders'.format(PORT))
            self.last_tick_took = get_tick()
            if resp.status_code == 200:
                tender = resp.json()
                if len(tender) > 0:
                    newOffer = self.tenderOffer(tender)

                    # if have pending position to unwind, be more generous with the price
                    if (newOffer.action == "BUY" and POSITION_MANAGER.get_position("RITC") <
                            -POSITION_MANAGER.trade_position_limit):
                        MARKETBOOK.updateMarketBook("RITC")
                        if newOffer.price < MARKETBOOK.get_mid_price("RITC", vwap=True):
                            self.take_offer_and_hedge(newOffer)
                    if (newOffer.action == "SELL" and POSITION_MANAGER.get_position("RITC") >
                            POSITION_MANAGER.trade_position_limit):
                        MARKETBOOK.updateMarketBook("RITC")
                        if newOffer.price > MARKETBOOK.get_mid_price("RITC", vwap=True):
                            self.take_offer_and_hedge(newOffer)
                    # if receive offer but no opposite position to unwind
                    # evaluate the profit and consider front run
                    else:
                        if self.evaluate_profitable_before_take(newOffer):
                            self.state = "FRONTRUN"
                            self.front_run(newOffer)
                            self.offer = newOffer

        # handle offers on hand and front run
        if self.state == "FRONTRUN" and get_tick() \
                - self.last_tick_took >= self.wait_tick:
            # a lot of things going on here!
            # will cancel orders or take positions on opp side if market moves against us
            self.handle_offer_after_front_run(self.offer)

            # reset the state
            self.state = "NOOFFER"
            self.offer = None
            self.last_tick_took = None

    def take_offer_and_hedge(self, offer):
        resp = SESSION.post('http://localhost:{}/v1/tenders?id={}'.format(
            PORT, self.offer.tender_id))
        if resp.status_code == 200:
            print("accepted tender {}!".format(self.offer))
        MARKETBOOK.hedge_usd(offer.price * offer.quantity, offer.action)

    # update orderBook when calling this function instead of handle_offer()!
    # logic: profitable if:
    # 1. spread is wide enough (i.e. vwap_mid + trailing_vol * multiplier)
    # 2. liquidity is sufficient
    def evaluate_profitable_before_take(self, offer):
        MARKETBOOK.updateMarketBook("RITC")
        # by default 20 data points
        trailing_stat = MARKETBOOK.get_RITC_trailing_stat()
        profit_flag = False
        # if data is not enough, be aggressive
        if trailing_stat is None:
            return True
        # if the agent wants to sell to you
        if offer.action == 'BUY':
            ideal_bid_price = trailing_stat["vwap_mid_avg"] - \
                trailing_stat["vwap_mid_vol"] * self.volatility_mult_before
            ask_volume = MARKETBOOK.volume_ask_arr[-1]
            profit_flag = (offer.price <= ideal_bid_price and
                           offer.quantity <= ask_volume * self.volume_threshold)
        else:
            ideal_ask_price = trailing_stat["vwap_mid_avg"] + \
                trailing_stat["vwap_mid_vol"] * self.volatility_mult_before
            bid_volume = MARKETBOOK.volume_bid_arr[-1]
            profit_flag = (offer.price >= ideal_ask_price and
                           offer.quantity <= bid_volume * self.volume_threshold)
        return profit_flag

    def front_run(self, offer):
        # TODO: params for front run!
        POSITION_MANAGER.updatePositionBook("RITC")
        size_if_take = offer.quantity + POSITION_MANAGER.get_position("RITC")
        # if the position is way too large, try front run more
        if size_if_take > POSITION_MANAGER.tender_position_limit:
            tgt_quantity = int(offer.quantity / 2)
        elif size_if_take > POSITION_MANAGER.trade_position_limit:
            tgt_quantity = int(offer.quantity / 3)
        else:
            tgt_quantity = int(offer.quantity / 4)
        if offer.action == "BUY":
            POSITION_MANAGER.add_split_orders("RITC", sell=True,
                                              num_splits=3, order_size=tgt_quantity)
        else:
            POSITION_MANAGER.add_split_orders("RITC", sell=False,
                                              num_splits=3, order_size=tgt_quantity)

    # update both market book and trader book when calling this!
    # evaluate profitable:
    #       1. reasonable price
    #       2. we don't have tons of open orders on the other side
    #       3. unwind it ASAP
    # if violate:
    #       1. if no liquidity, decline offer and cancel open orders
    #       2. BUT if the price is moving against us,
    #       take advantage of this by selling more!
    def handle_offer_after_front_run(self, offer):
        # params for evaluating liquidity
        # TODO: params
        move_against_threshold = .1

        MARKETBOOK.updateMarketBook("RITC")
        TRADERBOOK.updateTraderBook("RITC", "price", aggregate=True)
        trailing_stat = MARKETBOOK.get_RITC_trailing_stat()

        if offer.action == "BUY":
            unfilled_orders = TRADERBOOK.get_trader_volume(
                "RITC", "SELL", price_threshold=.05)
            ideal_bid_price = trailing_stat["vwap_mid_avg"] - \
                trailing_stat["vwap_mid_vol"] * self.volatility_mult_after

            if offer.price <= ideal_bid_price:
                # accept offer and unwind position
                self.take_offer_and_hedge(offer)
                POSITION_MANAGER.cancel_bad_orders("RITC", "SELL", spread_threshold=0.03)
                POSITION_MANAGER.unwind_position(sell=True)
            else:
                # if price is ok, but no liquidity, then do nothing
                # but if market is moving against us, take advantage of it!
                if unfilled_orders < offer.quantity * move_against_threshold:
                    # position managers figure out how much to trade
                    POSITION_MANAGER.add_split_orders("RITC", sell=True,
                                                      num_splits=4)
        else:
            unfilled_orders = TRADERBOOK.get_trader_volume(
                "RITC", "BUY", price_threshold=.05)
            ideal_bid_price = trailing_stat["vwap_mid_avg"] + \
                trailing_stat[
                "vwap_mid_vol"] * self.volatility_mult_after
            if offer.price >= ideal_bid_price and \
                    unfilled_orders <= offer.quantity * move_against_threshold:
                self.take_offer_and_hedge(offer)
                # unwind position
                POSITION_MANAGER.cancel_bad_orders("RITC", "BUY", spread_threshold=0.03)
                POSITION_MANAGER.unwind_position(sell=False)
            else:
                # if price is ok, but no liquidity, then do nothing
                # but if market is moving against us, take advantage of it!
                if unfilled_orders < offer.quantity * move_against_threshold:
                    POSITION_MANAGER.add_split_orders("RITC", sell=False, num_splits=4)


# Main Function
API_KEY = {'X-API-key': 'FWJX3L79'}
PORT = "9999"

# use global variable in all functions
with requests.Session() as SESSION:
    SESSION.headers.update(API_KEY)
MARKETBOOK = OrderBook("MARKET", size=5)
TRADERBOOK = OrderBook("TRADER", size=5)
POSITION_MANAGER = PositionManager()
TENDER = TenderManager(wait_tick=3)

while not check_case_status():
    pass

while get_tick() < 300:
    MARKETBOOK.updateAllMarketBook()
    TRADERBOOK.updateTraderBook("RITC", orderBy="price")
    POSITION_MANAGER.updatePositionBook("RITC")

    # Tender Offer First
    TENDER.handle_offer()
    # check if need to unwind position
    curr_position = POSITION_MANAGER.get_position("RITC")

    # Market Making
    # use VB and FB as signals
    # Position Manager sizes the trade
    # TODO: params
    VB = FB = 0
    if MARKETBOOK.RITC_record_count > 0:
        VB = MARKETBOOK.get_vol_imbalance()
        if MARKETBOOK.RITC_record_count > 1:
            FB = MARKETBOOK.get_flow_imbalance()

    if curr_position < POSITION_MANAGER.ideal_position_limit:
        if VB > 0.3 or FB > 5000:
            # will automatically hedge USD
            # cancel Sell orders and submit new Buy orders
            POSITION_MANAGER.cancel_bad_orders(
                "RITC", sell=True, close_to_best=True, spread_threshold=0.03)
            POSITION_MANAGER.add_split_orders("RITC", num_splits=3, sell=False)

    if curr_position > -POSITION_MANAGER.ideal_position_limit:
        if VB < -0.3 or FB < -5000:
            POSITION_MANAGER.cancel_bad_orders(
                "RITC", sell=False, close_to_best=True, spread_threshold=0.03)
            POSITION_MANAGER.add_split_orders("RITC", num_splits=3, sell=True)