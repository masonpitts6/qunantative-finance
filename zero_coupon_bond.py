import time_value as tv


class ZeroCouponBond:

    def __init__(self, principal, maturity, interest_rate):
        # Par value of the bond that will be paid at maturity
        self.principal = principal
        # Periods until the bond matures
        self.maturity = maturity
        # Market interest rate as float that converted to match the periods used to measure maturity e.g. semi-annual
        # or annual
        self.interest_rate = interest_rate

    def calculate_price(self):
        return tv.present_value_discrete(principal=self.principal, rate=self.interest_rate, periods=self.maturity)

if __name__ == '__main__':
    bond = ZeroCouponBond(principal=1000, maturity=2, interest_rate=0.04)
    print(f"Price of the bond: {bond.calculate_price():.2f}")